# Memristor Multimodal Modeling

This repository contains the code used in the paper:
> "Cross-modal prediction of memristor symmetry from EIS and SEM features" (2025).

## Contents
- DC symmetry metric calculation (`calculate_scores_guarded`)
- EIS feature extraction
- SEM morphological features
- Leave-One-Wafer-Out cross-validation with wafer-offset correction

## Usage
```bash
pip install -r requirements.txt
python main.py
