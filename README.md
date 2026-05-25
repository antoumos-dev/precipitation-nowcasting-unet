# Precipitation Nowcasting with U-Net

Short-term precipitation forecasting with deep learning — 10/20/30-minute radar extrapolation for Switzerland, trained on MeteoSwiss radar composites across a 3.5-year dataset over the period 2020–2023.

---

## Problem

Predicting where and how much it will rain 10–30 minutes ahead is a core challenge in operational meteorology. Classical optical-flow methods (e.g. pysteps) extrapolate current patterns but struggle with convective initiation and decay. This project trains a U-Net to learn the mapping directly from recent radar frames to multiple future frames simultaneously.

Convolutional architectures are well-suited for this task because radar composites are inherently spatial: precipitation patterns have local structure and translational regularity that convolutions can exploit efficiently. The U-Net combines an encoder branch — which progressively reduces spatial resolution while increasing feature depth to capture large-scale patterns — with a decoder branch that restores spatial resolution using skip connections from the encoder.

---

## Approach

| Component | Choice |
|---|---|
| Architecture | Multi-output U-Net — shared encoder, three parallel decoder branches (+10, +20, +30 min) |
| Input | 3 radar frames (log1p-scaled reflectivity), 501×371 px |
| Output | 3 predicted frames at +10, +20, +30 min |
| Loss | Lead-time weighted L1 + Log Spectral Distance (LSD) |
| Lead weights | ×1 / ×2 / ×3 for +10 / +20 / +30 min |
| Optimizer | AdamW, lr=1e-4, weight decay=1e-4, ReduceLROnPlateau (factor=0.5, patience=5) |
| Regularisation | BatchNorm + Dropout2d (0.1) + early stopping (patience=15) |

The model is trained in log space to stabilize the heavy-tailed precipitation distribution, but weights the loss in physical space to ensure high-intensity events drive the gradients.

The combined loss is defined as:

```
L = mean( w_lead × (1 + R) × |pred - target| ) + λ × LSD(pred, target)
```

where pred and target are in log1p(mm/10min) space, R = expm1(target) converts back to rain rate in mm/10min, w_lead ∈ {1, 2, 3} progressively upweights longer lead times, and λ = 0.01 scales the Log Spectral Distance term. The LSD term penalises blurriness by comparing predicted and observed power spectra, counteracting the spatial smoothing that pixel-wise losses tend to produce.

### Temporal consistency
The three decoder branches are connected via cross-connections at 64×48 resolution: +10 min decoder features are injected into the +20 min decoder, and +20 min features into the +30 min decoder. This encourages the model to produce temporally coherent predictions at the mesoscale.

Temporal autocorrelation diagnostics (spatial correlation of each predicted frame with t0) show that the model predictions decay more slowly than observed — the model tends to predict fields that remain too similar to the current state, particularly at +20 and +30 min. This is a known limitation of pixel-wise loss functions: without explicit motion information, the model hedges by producing a smoothed version of t0 rather than extrapolating the precipitation field forward. A ConvLSTM-based temporal encoder is under development to address this.

---

## Repository structure

```
├── notebooks
    ├── nowcast_01_preprocessing.ipynb   # raw radar → npz samples
    ├── nowcast_02_enrich_split.ipynb    # temporal train/val/test split + metadata
    └── nowcast_03_train.ipynb           # initial prototype (single-GPU, small data)
├── nowcast_04_train.py          # full training script (SLURM)
├── nowcast_05_test.py               # inference + metrics on held-out test set
├── radar_data/                          # raw radar data 
│   ├── training_samples_meta.csv        # event metadata
│   └── training_samples_meta_enriched.csv  # metadata with train/val/test split
├── scripts/
│   ├── run_train.sh                 # SLURM job for training
│   └── run_test.sh                  # SLURM job for evaluation
├── figures/                         # sample output plots 
│ 
└── utils/                           # helpers to be added
```

---

## Results

Evaluation on held-out test set (+30 min lead). Current configuration: **lead-time weighted L1 + Log Spectral Distance (LSD)** loss. The rain-rate penalty (`weight = 1 + R`) upweights heavy precipitation pixels; lead-time weights (×1/×2/×3) give progressively stronger gradient signal to longer leads; and the LSD term penalises spectral blurriness. Switching to plain MSE produced blurrier predictions and worse skill scores at moderate-to-high intensities. Persistence = last input frame held constant as the +30 min forecast.

| Metric | U-Net | Persistence |
|---|---|---|
| MAE (mm/10min) | **0.05** | 0.08 |
| RMSE (mm/10min) | **0.30** | 0.49 |
| CSI @ 0.1 mm/10min | **0.64** | 0.47 |
| CSI @ 0.5 mm/10min | **0.41** | 0.24 |
| CSI @ 1.0 mm/10min | **0.29** | 0.13 |
| FSS @ 0.1 mm/10min, 8 px | **0.88** | 0.76 |
| FSS @ 0.1 mm/10min, 32 px | **0.94** | 0.90 |
| FSS @ 0.5 mm/10min, 8 px | **0.59** | 0.52 |
| FSS @ 0.5 mm/10min, 32 px | 0.67 | **0.76** |
| FSS @ 1.0 mm/10min, 8 px | **0.41** | 0.35 |
| FSS @ 1.0 mm/10min, 32 px | 0.49 | **0.61** |

Results show improvement over the previous single-output baseline: RMSE reduced from 0.33 → 0.30, and FSS at high intensity/large scale improved (FSS @ 1.0mm, 32px: 0.47 → 0.49). The model still loses to persistence at large spatial scales and high thresholds (FSS @ 0.5mm and 1.0mm, 32px), consistent with spatial displacement errors in heavy precipitation cells at longer lead times.

---

## Quickstart

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate nowprecip
```

### 2. Prepare data

Download `training_data.npz` from the Zenodo repository (see **Data** section) and place it in `radar_data/` alongside the metadata files already included in this repo:
- `training_data.npz` — input/output radar stacks (from Zenodo)
- `training_samples_meta_enriched.csv` — metadata with `split` column (included)

See `nowcast_01_preprocessing.ipynb` and `nowcast_02_enrich_split.ipynb` for how the npz was generated from raw MeteoSwiss data.

### 3. Train

Edit `RUN_NAME` in `nowcast_04_train.py`, then:

```bash
# Locally
python nowcast_04_train.py

# On a SLURM cluster (submit from project root)
sbatch scripts/run_train.sh
```

Checkpoints are saved to `checkpoints/best_model_<RUN_NAME>.pt`.

### 4. Evaluate

```bash
python nowcast_05_test.py   # RUN_NAME must match training
```

Metrics and plots are saved to `test_output/<RUN_NAME>/`.

---

## Environment

Tested on:
- Python 3.10
- PyTorch 2.x
- CUDA 12.x
- pysteps, numpy, pandas, matplotlib

---

## Data

The processed radar composite dataset (`training_data.npz`) is publicly available on Zenodo:

> **TODO: insert Zenodo DOI badge and link here**
> [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Source: MeteoSwiss radar composite data, processed for short-term precipitation nowcasting over Switzerland (2020–2023).

The event metadata and train/val/test split (`training_samples_meta.csv`, `training_samples_meta_enriched.csv`) are included in this repository to allow full reproducibility of the split.
