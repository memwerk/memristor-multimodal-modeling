#!/usr/bin/env python3
"""
Memristor multimodal modeling — LOO-W CV + wafer-offset + micro CIs.

Implements:
- Guarded s1 DC symmetry target.
- SEM image morphology + FFT features.
- EIS dark/UV features and physics-inspired combinations.
- Leave-One-Wafer-Out cross-validation (macro, per-wafer metrics).
- Micro pooled metrics with bootstrap CIs (over wafers).
- Wafer-offset modeling variants (RF, GPR, KRR).
- Strong baseline: wafer-median with batch→global fallback.
- Per-sample prediction CSVs and diagnostic plots.

Typical usage:
    python memristor_multimodal_loo_wafer.py \
        --file-path output_data_2025-08-06_16-58.h5 \
        --out-dir loo_wafer_results
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.fft import fft2
from scipy.stats import entropy, kurtosis, skew, spearmanr
from skimage.feature import local_binary_pattern
from skimage.measure import label, regionprops

# robust gray/grey compatibility
try:
    from skimage.feature import greycomatrix as _glcm_fn, greycoprops as _glcm_props
except Exception:  # pragma: no cover - depends on skimage version
    from skimage.feature import graycomatrix as _glcm_fn, graycoprops as _glcm_props

from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    ParameterGrid,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# plotting backend for headless environments
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
set_config(enable_metadata_routing=True)

# ============================================================
# (A) Target and data helpers
# ============================================================


def calculate_scores_guarded(
    data: np.ndarray,
    U_skip: float = 0.02,
    I_min: float = 1e-9,
    l: float = 0.2,
):
    """
    Compute guarded quarter-sweep symmetry score s1 for a single DC sweep.

    Parameters
    ----------
    data : array, shape (N, >=4)
        DC sweep data, columns must contain U in col 2, I in col 3.
    U_skip : float
        Voltage guard around 0 V.
    I_min : float
        Minimum absolute current used for guards and eps.
    l : float
        Trimming parameter in [0, 0.5).

    Returns
    -------
    float or None
        Symmetry score s1, or None if not enough valid pairs.
    """
    U = data[:, 2].astype(float)
    I = data[:, 3].astype(float)
    N = len(U)
    if N < 8:
        return None

    I = np.clip(I, -1e6, 1e6)
    mask_valid = (np.abs(U) >= U_skip) & (np.abs(I) >= I_min)
    if mask_valid.sum() < 10:
        return None

    eps = I_min
    denom = np.where(np.abs(I) < eps, np.sign(I) * eps, I)
    R = U / denom

    lo, hi = np.percentile(R[mask_valid], [2, 98])
    R = np.clip(np.clip(R, lo, hi), -1e6, 1e6)

    m = (N - 1) // 2
    k0 = int(l * (N - 1) / 4)
    k1 = (N - 1) // 4 - 1
    Ks: List[int] = []

    for k in range(k0, k1 + 1):
        a = (k, m - k)
        b = (N - 1 - k, m + k)
        if min(a + b) < 0 or max(a + b) >= N:
            continue
        if (
            mask_valid[a[0]]
            and mask_valid[a[1]]
            and mask_valid[b[0]]
            and mask_valid[b[1]]
        ):
            Ks.append(k)

    if len(Ks) < 5:
        return None

    gamma = 2.0 / ((1.0 - l) * (N - 1))
    s1 = 0.0
    for k in Ks:
        d1 = R[m - k] if np.abs(R[m - k]) > eps else np.sign(R[m - k]) * eps
        d2 = R[m + k] if np.abs(R[m + k]) > eps else np.sign(R[m + k]) * eps
        r1 = R[k] / d1
        r2 = R[N - 1 - k] / d2
        s1 += (r1 + r2)

    return gamma * s1


def extract_all_image_features(image: np.ndarray, prefix: str = "img") -> Dict[str, float]:
    """Extract intensity, texture, morphology, and FFT-based features from a SEM image."""
    features: Dict[str, float] = {}
    image = image.astype(np.uint8)

    # basic stats
    features[f"{prefix}_mean"] = float(np.mean(image))
    features[f"{prefix}_std"] = float(np.std(image))
    features[f"{prefix}_skew"] = float(skew(image.ravel()))
    features[f"{prefix}_kurtosis"] = float(kurtosis(image.ravel()))

    # entropy
    hist = np.histogram(image.ravel(), bins=256, range=(0, 255), density=True)[0] + 1e-8
    features[f"{prefix}_entropy"] = float(entropy(hist))

    # GLCM texture
    glcm = _glcm_fn(
        image,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True,
    )
    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]:
        features[f"{prefix}_{prop}"] = float(_glcm_props(glcm, prop)[0, 0])

    # LBP histogram
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    for i, val in enumerate(lbp_hist):
        features[f"{prefix}_lbp_{i}"] = float(val)

    # morphology via adaptive threshold
    try:
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            10,
        )
    except cv2.error:
        _, thresh = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    labeled = label(thresh)
    regions = regionprops(labeled)

    features[f"{prefix}_particle_count"] = float(len(regions))
    areas, perimeters, solidities, eccentricities, aratios, circularities = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for region in regions:
        if region.area < 5:
            continue
        areas.append(region.area)
        perimeters.append(region.perimeter)
        solidities.append(region.solidity)
        eccentricities.append(region.eccentricity)
        aratios.append(
            region.major_axis_length / region.minor_axis_length
            if region.minor_axis_length > 0
            else 0.0
        )
        circularities.append(
            4 * np.pi * region.area / (region.perimeter**2)
            if region.perimeter > 0
            else 0.0
        )

    if areas:
        features[f"{prefix}_area_mean"] = float(np.mean(areas))
        features[f"{prefix}_perimeter_mean"] = float(np.mean(perimeters))
        features[f"{prefix}_solidity_mean"] = float(np.mean(solidities))
        features[f"{prefix}_eccentricity_mean"] = float(np.mean(eccentricities))
        features[f"{prefix}_aspect_ratio_mean"] = float(np.mean(aratios))
        features[f"{prefix}_circularity_mean"] = float(np.mean(circularities))
    else:
        for stat in [
            "area",
            "perimeter",
            "solidity",
            "eccentricity",
            "aspect_ratio",
            "circularity",
        ]:
            features[f"{prefix}_{stat}_mean"] = 0.0

    # FFT energy split
    fft_img = np.abs(fft2(image))
    fft_img = np.fft.fftshift(fft_img)
    center = np.array(fft_img.shape) // 2
    low = fft_img[center[0] - 10 : center[0] + 10, center[1] - 10 : center[1] + 10]
    mask = np.ones_like(fft_img, dtype=bool)
    mask[center[0] - 10 : center[0] + 10, center[1] - 10 : center[1] + 10] = False
    high = fft_img[mask]

    features[f"{prefix}_fft_low_energy"] = float(np.sum(low))
    features[f"{prefix}_fft_high_energy"] = float(np.sum(high))
    features[f"{prefix}_fft_ratio"] = (
        features[f"{prefix}_fft_high_energy"]
        / (features[f"{prefix}_fft_low_energy"] + 1e-8)
    )

    return features


def build_full_feature_dataframe(
    subtracted_images: Dict[str, np.ndarray],
    labels: np.ndarray,
    meta_list: List[Dict[str, object]],
) -> pd.DataFrame:
    """Build a DataFrame with SEM features, target scores, and metadata."""
    rows: List[Dict[str, object]] = []
    keys = list(subtracted_images.keys())
    for idx, key in enumerate(keys):
        img = subtracted_images[key]
        row: Dict[str, object] = {"key": key, "score": float(labels[idx])}
        row.update(extract_all_image_features(img, prefix="subtracted"))
        row.update(meta_list[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def align_images(reference: np.ndarray, to_align: np.ndarray) -> np.ndarray:
    """Align SEM images using ORB keypoints + homography. Falls back to original."""
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(reference, None)
    kp2, des2 = orb.detectAndCompute(to_align, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return to_align

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    if len(matches) < 4:
        return to_align

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    if H is None:
        return to_align
    return cv2.warpPerspective(to_align, H, (reference.shape[1], reference.shape[0]))


def extract_eis_features(eis_data: np.ndarray) -> Dict[str, float]:
    """Extract basic EIS features from a single EIS sweep (freq, |Z|, phase, R_p, C_p)."""
    freq = eis_data[:, 0]
    mag_z = eis_data[:, 5]
    phase_z = np.deg2rad(eis_data[:, 6])
    r_p = eis_data[:, 7]
    c_p = eis_data[:, 8]
    logf = np.log10(freq + 1e-6)

    return {
        "mag_z_mean": float(np.mean(mag_z)),
        "mag_z_slope": float(np.polyfit(logf, np.log10(mag_z + 1e-6), 1)[0]),
        "phase_z_mean": float(np.mean(phase_z)),
        "phase_z_slope": float(np.polyfit(logf, phase_z, 1)[0]),
        "mag_z_min": float(np.min(mag_z)),
        "mag_z_max": float(np.max(mag_z)),
        "r_p_mean": float(np.mean(r_p)),
        "r_p_min": float(np.min(r_p)),
        "r_p_max": float(np.max(r_p)),
        "c_p_mean": float(np.mean(c_p)),
        "c_p_min": float(np.min(c_p)),
        "c_p_max": float(np.max(c_p)),
    }


def extract_eis_combined_per_device(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    Aggregate EIS features per device, separately for dark and UV,
    and compute dark/UV deltas.
    """
    eis_combined: Dict[str, Dict[str, float]] = {}

    with h5py.File(file_path, "r") as h5_file:

        def process_group(name: str, group: h5py.Group):
            if "02eis" not in group or not isinstance(group["02eis"], h5py.Group):
                return

            eis_group = group["02eis"]
            if "eis_dark" not in eis_group or "eis_UV" not in eis_group:
                return

            dark_group, uv_group = eis_group["eis_dark"], eis_group["eis_UV"]
            dark_feats, uv_feats = [], []

            for _, ds in dark_group.items():
                if isinstance(ds, h5py.Dataset):
                    data = ds[()]
                    if data.shape[1] >= 9:
                        dark_feats.append(extract_eis_features(data))

            for _, ds in uv_group.items():
                if isinstance(ds, h5py.Dataset):
                    data = ds[()]
                    if data.shape[1] >= 9:
                        uv_feats.append(extract_eis_features(data))

            if not dark_feats and not uv_feats:
                return

            comb: Dict[str, float] = {}
            if dark_feats:
                dark_avg = {
                    f"{k}_DARK": float(np.mean([d[k] for d in dark_feats]))
                    for k in dark_feats[0]
                }
                comb.update(dark_avg)

            if uv_feats:
                uv_avg = {
                    f"{k}_UV": float(np.mean([d[k] for d in uv_feats]))
                    for k in uv_feats[0]
                }
                comb.update(uv_avg)

            if dark_feats and uv_feats:
                for dk in list(dark_avg.keys()):
                    k = dk.replace("_DARK", "")
                    comb[f"{k}_delta"] = float(
                        comb.get(f"{k}_UV", np.nan) - comb.get(f"{k}_DARK", np.nan)
                    )

            eis_combined[name] = comb

        h5_file.visititems(
            lambda name, obj: process_group(name, obj) if isinstance(obj, h5py.Group) else None
        )

    return eis_combined


def extract_all_data_with_eis(file_path: str):
    """
    Extract:
    - DC symmetry labels (guarded s1, then averaged per device).
    - SEM 01/03 aligned and subtracted images (per magnification).
    - EIS features (dark/UV).

    Returns
    -------
    subtracted_images : dict[key, np.ndarray]
    original_images_01 : dict[key, np.ndarray]
    original_images_03 : dict[key, np.ndarray]
    labels : np.ndarray
    meta_list : list of dict
    """
    subtracted_images: Dict[str, np.ndarray] = {}
    original_images_01: Dict[str, np.ndarray] = {}
    original_images_03: Dict[str, np.ndarray] = {}
    meta_list: List[Dict[str, object]] = []
    labels: List[float] = []

    def get_mag(filename: str):
        for m in ["10k", "5k", "1k"]:
            if m in filename.lower():
                return m
        return None

    eis_combined = extract_eis_combined_per_device(file_path)

    with h5py.File(file_path, "r") as h5_file:

        def process_group(group_name: str, group_obj: h5py.Group):
            # require SEM (01, 03) and DC
            if not all(k in group_obj and len(group_obj[k]) > 0 for k in ["01sem", "03sem", "04dc"]):
                return

            all_scores: List[float] = []
            meta_entry: Dict[str, object] | None = None

            # --- DC: compute s1 scores ---
            for dataset_name, dataset in group_obj["04dc"].items():
                if not isinstance(dataset, h5py.Dataset):
                    continue
                parts = dataset_name.split("_")
                if len(parts) < 6 or parts[0] in ["26", "32"]:
                    # skip guard/degenerate sets
                    continue

                data_04dc = dataset[()]
                s1 = calculate_scores_guarded(
                    data_04dc,
                    U_skip=0.02,
                    I_min=1e-9,
                    l=0.2,
                )
                if s1 is None:
                    continue

                all_scores.append(s1)
                if meta_entry is None:
                    meta_entry = {
                        "04dc_file": dataset_name,
                        "batch": parts[-3],
                        "wafer": parts[-2],
                        "die": parts[-1],
                    }

            if not all_scores or meta_entry is None:
                return

            avg_score = float(np.mean(all_scores))
            if group_name in eis_combined:
                meta_entry.update(eis_combined[group_name])

            # --- SEM images: 01 vs 03, per magnification ---
            imgs_01 = {
                get_mag(n): ds
                for n, ds in group_obj["01sem"].items()
                if isinstance(ds, h5py.Dataset) and get_mag(n) is not None
            }
            imgs_03 = {
                get_mag(n): ds
                for n, ds in group_obj["03sem"].items()
                if isinstance(ds, h5py.Dataset) and get_mag(n) is not None
            }

            for mag in imgs_01:
                if mag not in imgs_03:
                    continue

                img_01 = imgs_01[mag][()].astype(np.uint8)
                img_03 = imgs_03[mag][()].astype(np.uint8)

                h, w = min(img_01.shape[0], img_03.shape[0]), min(
                    img_01.shape[1],
                    img_03.shape[1],
                )
                img_01_c = img_01[:h, :w]
                img_03_c = img_03[:h, :w]
                aligned_03 = align_images(img_01_c, img_03_c)

                s = np.stack(
                    [img_01_c.astype(np.float32), aligned_03.astype(np.float32)],
                    axis=0,
                )
                mean, std = s.mean(), s.std() if s.std() > 0 else 1.0
                d = np.abs((s[0] - mean) / std - (s[1] - mean) / std)
                d = ((d - d.min()) / (d.max() - d.min() + 1e-8) * 255).astype(np.uint8)

                key = f"{group_name}/{mag}"
                subtracted_images[key] = d
                original_images_01[key] = img_01_c
                original_images_03[key] = aligned_03
                meta_list.append(meta_entry.copy())
                labels.append(avg_score)

        h5_file.visititems(
            lambda name, obj: process_group(name, obj) if isinstance(obj, h5py.Group) else None
        )

    return (
        subtracted_images,
        original_images_01,
        original_images_03,
        np.array(labels, dtype=float),
        meta_list,
    )


# ============================================================
# (B) Physics features + model zoo + preprocessing
# ============================================================


def _first_present(df: pd.DataFrame, names: List[str], default=np.nan) -> pd.Series:
    """Return the first present column from names, or a constant series."""
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index, dtype="float64")


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EIS-derived physics features and SEM aggregations."""
    out = df.copy()

    rp_dark = _first_present(out, ["r_p_mean_DARK", "r_p_DARK", "r_p_mean_dark", "r_p_mean"])
    cp_dark = _first_present(out, ["c_p_mean_DARK", "c_p_DARK", "c_p_mean_dark", "c_p_mean"])
    rp_uv = _first_present(out, ["r_p_mean_UV", "r_p_UV"])
    cp_uv = _first_present(out, ["c_p_mean_UV", "c_p_UV"])

    out["tau_dark"] = rp_dark * cp_dark
    out["tau_uv"] = rp_uv * cp_uv
    out["tau_delta"] = out["tau_uv"] - out["tau_dark"]

    ph_dark = _first_present(
        out,
        ["phase_z_mean_DARK", "phase_z_mean_dark", "phase_z_mean"],
    )
    ph_uv = _first_present(out, ["phase_z_mean_UV"])
    sl_mag_dark = _first_present(out, ["mag_z_slope_DARK", "mag_z_slope"])
    sl_mag_uv = _first_present(out, ["mag_z_slope_UV"])
    sl_ph_dark = _first_present(out, ["phase_z_slope_DARK", "phase_z_slope"])
    sl_ph_uv = _first_present(out, ["phase_z_slope_UV"])

    out["phase_delta"] = ph_uv - ph_dark
    out["mag_slope_delta"] = sl_mag_uv - sl_mag_dark
    out["phase_slope_delta"] = sl_ph_uv - sl_ph_dark

    out["debye_dev_dark"] = (sl_mag_dark + 1.0).abs()
    out["debye_dev_uv"] = (sl_mag_uv + 1.0).abs()

    out["Qproxy_dark"] = ph_dark.abs()
    out["Qproxy_uv"] = ph_uv.abs()
    out["Qproxy_delta"] = out["Qproxy_uv"] - out["Qproxy_dark"]

    out["fft_fine_to_coarse"] = _first_present(
        out,
        ["subtracted_fft_ratio", "subtracted_fft_high_energy"],
    )
    out["morph_count"] = _first_present(out, ["subtracted_particle_count"])

    if "QDC" not in out.columns:
        out["QDC"] = np.nan

    return out


def robust_filter_targets(
    y: np.ndarray,
    max_z: float = 4.0,
    p99_5: bool = True,
) -> np.ndarray:
    """Robustly filter extreme outliers using MAD & optional 99.5% upper clip."""
    y = np.asarray(y, float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med)) + 1e-12
    z = 0.6745 * (y - med) / mad
    keep = np.abs(z) <= max_z
    if p99_5 and keep.any():
        t = np.nanpercentile(y[keep], 99.5)
        keep &= y <= t
    return keep


def assemble_xy_groups(
    df: pd.DataFrame,
    target_col: str = "score",
    drop_non_numeric: bool = True,
):
    """Add physics features and assemble (X, y, groups, wafers, batches, feature_names, df_aug)."""
    df2 = add_physics_features(df)

    for req in ["batch", "wafer", "die"]:
        if req not in df2.columns:
            df2[req] = "UNK"

    df2["group_id"] = (
        df2["batch"].astype(str)
        + "/"
        + df2["wafer"].astype(str)
        + "/"
        + df2["die"].astype(str)
    )

    ignore = {target_col, "key", "group_id", "04dc_file", "batch", "wafer", "die"}
    if drop_non_numeric:
        feat_cols = [
            c
            for c in df2.columns
            if c not in ignore and np.issubdtype(df2[c].dtype, np.number)
        ]
    else:
        feat_cols = [c for c in df2.columns if c not in ignore]

    X = df2[feat_cols].copy()
    y = df2[target_col].astype(float).values
    groups = df2["group_id"].values
    wafers = df2["wafer"].astype(str).values
    batches = df2["batch"].astype(str).values

    return X, y, groups, wafers, batches, feat_cols, df2


# ---- Preprocessing and models ----


class SignFlip(BaseEstimator, TransformerMixin):
    """Optionally flip sign of selected columns to enforce monotonicity priors."""

    def __init__(self, flip_indices: List[int] | None = None):
        self.flip_indices = flip_indices or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.size == 0 or not self.flip_indices:
            return X
        X2 = X.copy()
        X2[:, self.flip_indices] *= -1.0
        return X2


def make_preprocessor(num_cols: List[str], use_selector: bool = True) -> ColumnTransformer:
    """Build numeric preprocessor: impute → low-variance removal → scale → optional RF selector."""
    steps: List[Tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-6)),
        ("scale", StandardScaler()),
    ]
    if use_selector:
        steps.append(
            (
                "sfm",
                SelectFromModel(
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=-1,
                    ),
                    threshold="median",
                ),
            )
        )

    return ColumnTransformer(
        [("num", Pipeline(steps), num_cols)],
        remainder="drop",
    )


def _make_phys_lasso(
    num_cols: List[str],
    negative_expect: set | None = None,
    use_elasticnet: bool = False,
) -> Pipeline:
    """Lasso/ElasticNet with robust scaling and optional sign flipping for selected features."""
    negative_expect = negative_expect or set()
    flip_indices = [i for i, c in enumerate(num_cols) if c in negative_expect]

    num_steps: List[Tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold(1e-12)),
        ("scale", RobustScaler()),
    ]
    if flip_indices:
        num_steps.append(("flip", SignFlip(flip_indices=flip_indices)))

    if use_elasticnet:
        base = ElasticNet(
            alpha=0.1,
            l1_ratio=0.8,
            max_iter=50000,
            random_state=42,
        )
    else:
        base = Lasso(alpha=0.1, max_iter=50000, random_state=42)
        if hasattr(base, "positive"):
            base.set_params(positive=False)

    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("num", Pipeline(num_steps), num_cols)],
                    remainder="drop",
                ),
            ),
            ("lasso", base),
        ]
    )


# optional HGB vs GB fallback
try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HGB_AVAILABLE = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor

    _HGB_AVAILABLE = False


def make_models(
    num_cols: List[str],
    rf_params: Dict[str, object] | None = None,
    use_selector: bool = True,
):
    """Construct RF, GPR, HGB/GB, and KRR pipelines with a shared preprocessor."""
    pre = make_preprocessor(num_cols, use_selector=use_selector)

    rf_defaults = dict(
        n_estimators=800,
        max_features=0.4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    if rf_params:
        rf_defaults.update({k.replace("rf__", ""): v for k, v in rf_params.items()})

    rf = Pipeline(
        [
            ("prep", pre),
            ("rf", RandomForestRegressor(**rf_defaults)),
        ]
    )

    if _HGB_AVAILABLE:
        hgb_est = HistGradientBoostingRegressor(
            learning_rate=0.06,
            max_iter=800,
            l2_regularization=1e-3,
            random_state=42,
        )
    else:
        from sklearn.ensemble import GradientBoostingRegressor

        hgb_est = GradientBoostingRegressor(
            learning_rate=0.06,
            n_estimators=800,
            random_state=42,
        )

    hgb = Pipeline([("prep", pre), ("hgb", hgb_est)])

    gpr_kernel = ConstantKernel(1.0) * (RBF(1.0) + RationalQuadratic(0.5, 1.0)) + WhiteKernel(
        1e-2
    )
    gpr = Pipeline(
        [
            ("prep", pre),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=gpr_kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=2,
                    random_state=42,
                ),
            ),
        ]
    )

    krr = Pipeline(
        [
            ("prep", pre),
            ("krr", KernelRidge(alpha=5e-3, kernel="rbf")),
        ]
    )

    return {
        "RandomForest": rf,
        "GaussianProcess": gpr,
        "HGB_or_GB": hgb,
        "KernelRidge_RBF": krr,
    }


def tune_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    """Group-aware RF hyperparameter tuning with GroupKFold CV."""
    base = Pipeline(
        [
            ("prep", make_preprocessor(X.columns.tolist(), use_selector=True)),
            ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    grid = ParameterGrid(
        {
            "rf__n_estimators": [600, 1000],
            "rf__max_features": [0.3, 0.4, 0.5],
            "rf__min_samples_leaf": [1, 2, 4],
        }
    )
    gkf = GroupKFold(n_splits=min(5, np.unique(groups).shape[0]))
    best, best_rmse = None, 1e9

    for params in grid:
        base.set_params(**params)
        rmses: List[float] = []
        for tr, te in gkf.split(X, y, groups):
            Xt, yt = X.iloc[tr], y[tr]
            Xv, yv = X.iloc[te], y[te]
            base.fit(Xt, yt)
            yp = base.predict(Xv)
            rmses.append(np.sqrt(mean_squared_error(yv, yp)))
        m = float(np.mean(rmses))
        if m < best_rmse:
            best_rmse, best = m, params.copy()

    return best


def tune_krr_grouped(
    X: pd.DataFrame,
    y: np.ndarray,
    wafers: np.ndarray,
):
    """Group-aware KRR tuning with GroupKFold CV."""
    pre = make_preprocessor(X.columns.tolist(), use_selector=True)
    krr_pipe = Pipeline(
        [
            ("prep", pre),
            ("krr", KernelRidge(kernel="rbf")),
        ]
    )

    param_grid = {
        "krr__alpha": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
        "krr__gamma": [
            "scale",
            1e-3,
            5e-3,
            1e-2,
            5e-2,
            1e-1,
            0.5,
            1.0,
        ],
    }

    gcv = GridSearchCV(
        krr_pipe,
        param_grid,
        cv=GroupKFold(n_splits=min(5, np.unique(wafers).shape[0])),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gcv.fit(X, y, groups=wafers)
    return gcv.best_estimator_, gcv.best_params_


# ============================================================
# (C) Wafer-offset helpers
# ============================================================


def _compute_train_offsets(
    y_train: np.ndarray,
    wafers_train: np.ndarray,
    batches_train: np.ndarray,
):
    df_off = pd.DataFrame({"y": y_train, "wafer": wafers_train, "batch": batches_train})
    wafer_med = df_off.groupby("wafer")["y"].median().to_dict()
    batch_med = df_off.groupby("batch")["y"].median().to_dict()
    global_med = float(df_off["y"].median())
    wafer_to_batch = (
        df_off.drop_duplicates("wafer")[["wafer", "batch"]]
        .set_index("wafer")["batch"]
        .to_dict()
    )
    return wafer_med, batch_med, global_med, wafer_to_batch


def _offset_for_val_wafer(
    wafer: str,
    wafer_to_batch: Dict[str, str],
    wafer_med: Dict[str, float],
    batch_med: Dict[str, float],
    global_med: float,
) -> float:
    """Return wafer-specific baseline, falling back to batch and then global."""
    if wafer in wafer_med:
        return wafer_med[wafer]
    batch = wafer_to_batch.get(wafer, None)
    if batch is not None and batch in batch_med:
        return batch_med[batch]
    return global_med


# ============================================================
# (D) LOO-W evaluation (macro per-wafer metrics)
# ============================================================


@dataclass
class FoldRecord:
    wafer: str
    rmse: float
    mae: float
    nrmse: float
    r2: float
    spearman: float
    n_samples: int


@dataclass
class ModelSummary:
    model_name: str
    rmse_mean: float
    rmse_ci: Tuple[float, float]
    mae_mean: float
    mae_ci: Tuple[float, float]
    nrmse_mean: float
    nrmse_ci: Tuple[float, float]
    r2_mean: float
    r2_ci: Tuple[float, float]
    spearman_mean: float
    spearman_ci: Tuple[float, float]
    per_wafer: List[FoldRecord]


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray, nrmse_den: float):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nrmse = float(rmse / (nrmse_den + 1e-12))
    with np.errstate(invalid="ignore"):
        rho = spearmanr(y_true, y_pred).correlation
    rho = float(rho) if np.isfinite(rho) else 0.0
    r2 = float(r2_score(y_true, y_pred))
    return rmse, mae, nrmse, r2, rho


def _bootstrap_group_ci(
    df_per_group: pd.DataFrame,
    value_col: str,
    group_col: str,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap CI over groups (wafers)."""
    rng = np.random.default_rng(seed)
    groups = df_per_group[group_col].unique()
    B: List[float] = []

    for _ in range(n_boot):
        sample_groups = rng.choice(groups, size=len(groups), replace=True)
        B.append(
            float(
                df_per_group.set_index(group_col)
                .loc[sample_groups, value_col]
                .mean()
            )
        )

    lo = float(np.quantile(B, alpha / 2))
    hi = float(np.quantile(B, 1 - alpha / 2))
    return lo, hi


def loo_wafer_eval(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: np.ndarray,
    wafers: np.ndarray,
    batches: np.ndarray,
    use_wafer_offset_variant: bool,
    nrmse_denominator: float,
) -> List[ModelSummary]:
    """Leave-One-Wafer-Out macro evaluation with optional wafer-offset modeling."""
    logo = LeaveOneGroupOut()
    results: List[ModelSummary] = []

    splits = list(logo.split(X, y, groups=wafers))
    wafer_order = [np.unique(wafers[te])[0] for _, te in splits]

    for name, base_model in models.items():
        fold_rows: List[FoldRecord] = []

        for (tr, te), wafer in zip(splits, wafer_order):
            Xt, yt = X.iloc[tr], y[tr]
            Xv, yv = X.iloc[te], y[te]
            waf_tr, waf_te = wafers[tr], wafers[te]
            bat_tr, bat_te = batches[tr], batches[te]

            model = clone(base_model)

            if use_wafer_offset_variant:
                wafer_med, batch_med, gmed, wafer_to_batch_tr = _compute_train_offsets(
                    yt,
                    waf_tr,
                    bat_tr,
                )
                yt_resid = yt - np.vectorize(wafer_med.get)(waf_tr)
                model.fit(Xt, yt_resid)

                offset_val = _offset_for_val_wafer(
                    wafer=wafer,
                    wafer_to_batch={
                        **{w: b for w, b in zip(waf_te, bat_te)},
                        **{w: b for w, b in zip(waf_tr, bat_tr)},
                    },
                    wafer_med=wafer_med,
                    batch_med=batch_med,
                    global_med=gmed,
                )
                yv_pred = model.predict(Xv) + float(offset_val)
            else:
                model.fit(Xt, yt)
                yv_pred = model.predict(Xv)

            rmse, mae, nrmse, r2, rho = _metric_pack(yv, yv_pred, nrmse_denominator)
            fold_rows.append(
                FoldRecord(
                    wafer=wafer,
                    rmse=rmse,
                    mae=mae,
                    nrmse=nrmse,
                    r2=r2,
                    spearman=rho,
                    n_samples=len(te),
                )
            )

        dfw = pd.DataFrame([fr.__dict__ for fr in fold_rows])

        rmse_mean = float(dfw["rmse"].mean())
        rmse_ci = _bootstrap_group_ci(dfw, "rmse", "wafer")
        mae_mean = float(dfw["mae"].mean())
        mae_ci = _bootstrap_group_ci(dfw, "mae", "wafer")
        nrmse_mean = float(dfw["nrmse"].mean())
        nrmse_ci = _bootstrap_group_ci(dfw, "nrmse", "wafer")
        r2_mean = float(dfw["r2"].mean())
        r2_ci = _bootstrap_group_ci(dfw, "r2", "wafer")
        sp_mean = float(dfw["spearman"].mean())
        sp_ci = _bootstrap_group_ci(dfw, "spearman", "wafer")

        results.append(
            ModelSummary(
                model_name=name,
                rmse_mean=rmse_mean,
                rmse_ci=rmse_ci,
                mae_mean=mae_mean,
                mae_ci=mae_ci,
                nrmse_mean=nrmse_mean,
                nrmse_ci=nrmse_ci,
                r2_mean=r2_mean,
                r2_ci=r2_ci,
                spearman_mean=sp_mean,
                spearman_ci=sp_ci,
                per_wafer=fold_rows,
            )
        )

    return results


# ============================================================
# (E) Micro pooled metrics + CIs, baseline, predictions
# ============================================================


def _excel_col(idx: int) -> str:
    """0-based index -> A, B, ..., Z, AA, AB, ... (Excel-style)."""
    s = ""
    n = idx
    while True:
        n, r = divmod(n, 26)
        s = chr(ord("A") + r) + s
        if n == 0:
            break
        n -= 1
    return s


def make_wafer_letter_map(wafers: List[str]) -> Dict[str, str]:
    """Stable mapping from sorted wafer names to A, B, C, ..."""
    wafers_sorted = sorted(wafers)
    return {w: _excel_col(i) for i, w in enumerate(wafers_sorted)}


def compute_micro_metrics(y_true_all: np.ndarray, y_pred_all: np.ndarray):
    """Compute pooled RMSE, MAE, R², Spearman over all samples."""
    mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
    y_t = y_true_all[mask]
    y_p = y_pred_all[mask]
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae = float(mean_absolute_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    with np.errstate(invalid="ignore"):
        rho = spearmanr(y_t, y_p).correlation
    rho = float(rho) if np.isfinite(rho) else 0.0
    return rmse, mae, r2, rho


def bootstrap_micro_ci(
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Bootstrap micro metrics over wafers."""
    rng = np.random.default_rng(seed)
    dfp = pd.DataFrame({"y": y_true_all, "yp": y_pred_all, "wafer": groups})
    wafers = dfp["wafer"].unique()

    R_rmse: List[float] = []
    R_mae: List[float] = []
    R_r2: List[float] = []
    R_rho: List[float] = []

    for _ in range(n_boot):
        ws = rng.choice(wafers, size=len(wafers), replace=True)
        pick = pd.concat([dfp[dfp["wafer"] == w] for w in ws], ignore_index=True)
        rmse, mae, r2, rho = compute_micro_metrics(pick["y"].values, pick["yp"].values)
        R_rmse.append(rmse)
        R_mae.append(mae)
        R_r2.append(r2)
        R_rho.append(rho)

    def q(a: List[float]):
        return float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))

    return q(R_rmse), q(R_mae), q(R_r2), q(R_rho)


def rebuild_preds(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    wafers: np.ndarray,
    batches: np.ndarray,
    use_offset: bool,
) -> np.ndarray:
    """Rebuild LOO-W predictions for a model (with or without wafer-offset correction)."""
    yp = np.empty_like(y, float)
    yp[:] = np.nan
    logo = LeaveOneGroupOut()

    for tr, te in logo.split(X, y, groups=wafers):
        Xt, yt = X.iloc[tr], y[tr]
        Xv = X.iloc[te]
        waf_tr, waf_te = wafers[tr], wafers[te]
        bat_tr, bat_te = batches[tr], batches[te]
        m = clone(model)

        if use_offset:
            wafer_med, batch_med, gmed, _ = _compute_train_offsets(yt, waf_tr, bat_tr)
            yt_resid = yt - np.vectorize(wafer_med.get)(waf_tr)
            m.fit(Xt, yt_resid)
            wafer_val = np.unique(waf_te)[0]
            offset_val = _offset_for_val_wafer(
                wafer_val,
                wafer_to_batch={
                    **{w: b for w, b in zip(waf_te, bat_te)},
                    **{w: b for w, b in zip(waf_tr, bat_tr)},
                },
                wafer_med=wafer_med,
                batch_med=batch_med,
                global_med=gmed,
            )
            yp[te] = m.predict(Xv) + float(offset_val)
        else:
            m.fit(Xt, yt)
            yp[te] = m.predict(Xv)

    return yp


# ============================================================
# (F) Plotting
# ============================================================


def plot_per_wafer_rmse(
    df_per_wafer: pd.DataFrame,
    title: str,
    out_path: str,
) -> None:
    df_sort = df_per_wafer.sort_values("rmse", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.bar(df_sort["wafer"].astype(str), df_sort["rmse"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter_y_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=18, alpha=0.7)
    lims = [
        min(np.min(y_true), np.min(y_pred)),
        max(np.max(y_true), np.max(y_pred)),
    ]
    plt.plot(lims, lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    rho = spearmanr(y_true, y_pred).correlation
    if np.isfinite(rho):
        plt.title(f"{title}\nSpearman ρ = {rho:.3f}")
    else:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_per_wafer_rmse_combined(
    summaries: List[ModelSummary],
    out_path: str,
    title: str = "Per-wafer RMSE — All Models",
) -> None:
    """
    Combined per-wafer RMSE plot across all models.

    Writes:
    - PNG bar chart
    - wide CSV table with wafer labels (A,B,...) as index
    - wafer_mapping.csv mapping labels back to wafer ids
    """
    # collect union of wafers and ordered model names (best overall first)
    model_order = [s.model_name for s in sorted(summaries, key=lambda z: z.rmse_mean)]
    wafers = sorted({fr.wafer for s in summaries for fr in s.per_wafer})

    # build wafer->letter mapping and save it
    letter_map = make_wafer_letter_map(wafers)
    map_csv = os.path.splitext(out_path)[0] + "_wafer_mapping.csv"
    pd.DataFrame(
        {"wafer": wafers, "label": [letter_map[w] for w in wafers]}
    ).to_csv(map_csv, index=False)

    W = len(wafers)
    M = len(model_order)
    # wide table: rows = wafers, cols = models
    mat = np.full((W, M), np.nan, float)
    for j, name in enumerate(model_order):
        s = next(x for x in summaries if x.model_name == name)
        lookup = {fr.wafer: fr.rmse for fr in s.per_wafer}
        for i, w in enumerate(wafers):
            if w in lookup:
                mat[i, j] = lookup[w]

    # save wide csv with lettered index
    df_wide = pd.DataFrame(
        mat,
        index=[letter_map[w] for w in wafers],
        columns=model_order,
    )
    wide_csv = os.path.splitext(out_path)[0] + ".csv"
    df_wide.to_csv(wide_csv, index_label="wafer_label")

    # plot grouped bars
    x = np.arange(W)
    barw = min(0.8 / max(M, 1), 0.18)
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * W), 6))

    for j, name in enumerate(model_order):
        offs = (j - (M - 1) / 2) * barw
        yj = mat[:, j]
        ax.bar(
            x + offs,
            np.nan_to_num(yj, nan=0.0),
            width=barw,
            label=name,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([letter_map[w] for w in wafers], rotation=0)
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    note = "Wafer labels: " + ", ".join(f"{letter_map[w]}={w}" for w in wafers)
    ax.text(
        0.01,
        -0.22,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        wrap=True,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# (G) Main entry point
# ============================================================


def main(file_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # === Data ===
    (
        sub_images,
        img01,
        img03,
        labels,
        meta_list,
    ) = extract_all_data_with_eis(file_path)

    df = build_full_feature_dataframe(sub_images, labels, meta_list)
    print(
        "y before filtering:",
        "min",
        float(np.min(df["score"])),
        "med",
        float(np.median(df["score"])),
        "p95",
        float(np.percentile(df["score"], 95)),
        "max",
        float(np.max(df["score"])),
    )

    keep = robust_filter_targets(df["score"].values, max_z=4.0, p99_5=True)
    df = df.loc[keep].reset_index(drop=True)
    print(
        "y after filtering:",
        "min",
        float(np.min(df["score"])),
        "med",
        float(np.median(df["score"])),
        "p95",
        float(np.percentile(df["score"], 95)),
        "max",
        float(np.max(df["score"])),
    )

    # Assemble
    X, y, groups, wafers, batches, feat_cols, df_aug = assemble_xy_groups(
        df,
        target_col="score",
    )
    nrmse_den = float(np.max(y) - np.min(y))  # range-based denominator
    p5, p95 = np.percentile(y, [5, 95])  # robust denominator for reporting

    # === Tuning (grouped CV) ===
    print("\nTuning models (grouped CV)...")
    rf_best = tune_random_forest(X, y, groups)
    krr_best_est, krr_best_params = tune_krr_grouped(X, y, wafers)
    print("Best RF params:", rf_best)
    print("Best KRR params:", krr_best_params)

    # Build model zoo
    models = make_models(feat_cols, rf_params=rf_best, use_selector=True)
    models["KernelRidge_RBF"] = krr_best_est  # tuned KRR

    # Standard models for LOO-W macro eval
    models_standard = {
        "RandomForest": models["RandomForest"],
        "GaussianProcess": models["GaussianProcess"],
        "KernelRidge_RBF_Tuned": models["KernelRidge_RBF"],
        "HGB": models["HGB_or_GB"],
    }

    # Wafer-offset variants (RF, GPR, KRR)
    models_wafer_offset = {
        "RandomForest_WaferOffset": models["RandomForest"],
        "GaussianProcess_WaferOffset": models["GaussianProcess"],
        "KernelRidge_RBF_Tuned_WaferOffset": models["KernelRidge_RBF"],
    }

    # === LOO-W MACRO (per-wafer) ===
    print("\n===== LOO-W (standard) =====")
    res_std = loo_wafer_eval(
        models_standard,
        X,
        y,
        wafers,
        batches,
        use_wafer_offset_variant=False,
        nrmse_denominator=nrmse_den,
    )

    print("\n===== LOO-W (wafer-offset variant) =====")
    res_off = loo_wafer_eval(
        models_wafer_offset,
        X,
        y,
        wafers,
        batches,
        use_wafer_offset_variant=True,
        nrmse_denominator=nrmse_den,
    )

    # Save macro summaries + plots
    all_summaries: List[Dict[str, object]] = []

    for res_list in [res_std, res_off]:
        for summ in res_list:
            dfw = pd.DataFrame([fr.__dict__ for fr in summ.per_wafer])
            dfw_path = os.path.join(out_dir, f"per_wafer_{summ.model_name}.csv")
            dfw.to_csv(dfw_path, index=False)

            all_summaries.append(
                {
                    "model": summ.model_name,
                    "rmse_mean": summ.rmse_mean,
                    "rmse_ci_lo": summ.rmse_ci[0],
                    "rmse_ci_hi": summ.rmse_ci[1],
                    "mae_mean": summ.mae_mean,
                    "mae_ci_lo": summ.mae_ci[0],
                    "mae_ci_hi": summ.mae_ci[1],
                    "nrmse_mean": summ.nrmse_mean,
                    "nrmse_ci_lo": summ.nrmse_ci[0],
                    "nrmse_ci_hi": summ.nrmse_ci[1],
                    "r2_mean": summ.r2_mean,
                    "r2_ci_lo": summ.r2_ci[0],
                    "r2_ci_hi": summ.r2_ci[1],
                    "spearman_mean": summ.spearman_mean,
                    "spearman_ci_lo": summ.spearman_ci[0],
                    "spearman_ci_hi": summ.spearman_ci[1],
                }
            )

    # Combined per-wafer RMSE plot across all models
    plot_per_wafer_rmse_combined(
        summaries=[*res_std, *res_off],
        out_path=os.path.join(out_dir, "per_wafer_rmse_ALL_MODELS.png"),
        title="Per-wafer RMSE — All Models (LOO-W)",
    )

    leaderboard_macro = pd.DataFrame(all_summaries).sort_values("rmse_mean")
    leaderboard_macro.to_csv(
        os.path.join(out_dir, "leaderboard_loo_wafer.csv"),
        index=False,
    )

    # === LOO-W MICRO (pooled predictions + CIs) ===
    logo = LeaveOneGroupOut()

    def wafer_median_baseline_preds() -> np.ndarray:
        yp = np.empty_like(y, float)
        yp[:] = np.nan
        for tr, te in logo.split(X, y, groups=wafers):
            waf_tr, bat_tr = wafers[tr], batches[tr]
            yt = y[tr]
            wafer_med, batch_med, gmed, _ = _compute_train_offsets(yt, waf_tr, bat_tr)
            wafer_val = np.unique(wafers[te])[0]
            yp[te] = _offset_for_val_wafer(
                wafer_val,
                wafer_to_batch={},
                wafer_med=wafer_med,
                batch_med=batch_med,
                global_med=gmed,
            )
        return yp

    def micro_block(
        models_dict: Dict[str, Pipeline],
        tag: str,
        use_offset: bool,
    ) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for name, mdl in models_dict.items():
            yp = rebuild_preds(mdl, X, y, wafers, batches, use_offset)
            # save per-sample predictions
            pd.DataFrame(
                {
                    "wafer": wafers,
                    "batch": batches,
                    "y": y,
                    "yp": yp,
                }
            ).to_csv(
                os.path.join(out_dir, f"preds_{name}.csv"),
                index=False,
            )

            rmse, mae, r2, rho = compute_micro_metrics(y, yp)
            (rmse_lo, rmse_hi), (mae_lo, mae_hi), (r2_lo, r2_hi), (rho_lo, rho_hi) = (
                bootstrap_micro_ci(y, yp, wafers)
            )

            rows.append(
                {
                    "model": f"{name}",
                    "tag": tag,
                    "rmse_micro": rmse,
                    "rmse_micro_lo": rmse_lo,
                    "rmse_micro_hi": rmse_hi,
                    "mae_micro": mae,
                    "mae_micro_lo": mae_lo,
                    "mae_micro_hi": mae_hi,
                    "r2_micro": r2,
                    "r2_micro_lo": r2_lo,
                    "r2_micro_hi": r2_hi,
                    "spearman_micro": rho,
                    "spearman_micro_lo": rho_lo,
                    "spearman_micro_hi": rho_hi,
                    "nrmse_micro_range": rmse / (np.max(y) - np.min(y) + 1e-12),
                    "nrmse_micro_p95p5": rmse / max(p95 - p5, 1e-12),
                }
            )
        return pd.DataFrame(rows)

    micro_std = micro_block(models_standard, "standard", use_offset=False)
    micro_off = micro_block(models_wafer_offset, "wafer_offset", use_offset=True)

    # baseline (wafer median)
    yp_baseline = wafer_median_baseline_preds()
    rmse_b, mae_b, r2_b, rho_b = compute_micro_metrics(y, yp_baseline)
    (
        (rmse_lo_b, rmse_hi_b),
        (mae_lo_b, mae_hi_b),
        (r2_lo_b, r2_hi_b),
        (rho_lo_b, rho_hi_b),
    ) = bootstrap_micro_ci(y, yp_baseline, wafers)

    baseline_row = pd.DataFrame(
        [
            {
                "model": "Baseline_WaferMedian",
                "tag": "baseline",
                "rmse_micro": rmse_b,
                "rmse_micro_lo": rmse_lo_b,
                "rmse_micro_hi": rmse_hi_b,
                "mae_micro": mae_b,
                "mae_micro_lo": mae_lo_b,
                "mae_micro_hi": mae_hi_b,
                "r2_micro": r2_b,
                "r2_micro_lo": r2_lo_b,
                "r2_micro_hi": r2_hi_b,
                "spearman_micro": rho_b,
                "spearman_micro_lo": rho_lo_b,
                "spearman_micro_hi": rho_hi_b,
                "nrmse_micro_range": rmse_b / (np.max(y) - np.min(y) + 1e-12),
                "nrmse_micro_p95p5": rmse_b / max(p95 - p5, 1e-12),
            }
        ]
    )

    leaderboard_micro = pd.concat(
        [micro_std, micro_off, baseline_row],
        ignore_index=True,
    )
    leaderboard_micro.to_csv(
        os.path.join(out_dir, "leaderboard_micro_loo_wafer.csv"),
        index=False,
    )

    # === Best model by macro RMSE: rebuild & scatter ===
    best_model_name = leaderboard_macro.iloc[0]["model"]
    print("\nBest (LOO-W macro) model by RMSE:", best_model_name)

    if best_model_name in models_standard:
        best_model = models_standard[best_model_name]
        use_offset_flag = False
    else:
        best_model = models_wafer_offset[best_model_name]
        use_offset_flag = True

    y_pred_all = rebuild_preds(best_model, X, y, wafers, batches, use_offset_flag)
    plot_scatter_y_vs_pred(
        y,
        y_pred_all,
        title=f"True vs Predicted — {best_model_name} (LOO-W)",
        out_path=os.path.join(out_dir, f"scatter_true_vs_pred_{best_model_name}.png"),
    )

    print(f"\nSaved artifacts in: {out_dir}")
    print(
        f"- Macro leaderboard: {os.path.join(out_dir, 'leaderboard_loo_wafer.csv')}"
    )
    print(
        f"- Micro leaderboard: {os.path.join(out_dir, 'leaderboard_micro_loo_wafer.csv')}"
    )
    print("- Per-wafer CSVs & plots, per-model prediction CSVs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Memristor multimodal modeling (LOO-W CV + wafer-offset + micro CIs)."
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default="output_data_2025-08-06_16-58.h5",
        help="Path to input HDF5 file with DC, EIS, and SEM data.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="loo_wafer_results",
        help="Output directory for CSVs and plots.",
    )
    args = parser.parse_args()

    main(file_path=args.file_path, out_dir=args.out_dir)
