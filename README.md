# Precipitation Nowcasting with U-Net

Short-term precipitation forecasting with deep learning — 30-minute radar extrapolation for Switzerland, trained on MeteoSwiss radar composites


---

## Problem

Predicting where and how much it will rain 30 minutes ahead is a core challenge in operational meteorology. Classical optical-flow methods (e.g. pysteps) extrapolate current patterns but struggle with convective initiation and decay. This project trains a U-Net to learn the mapping directly from recent radar frames to a future frame.

Convolutional architectures are well-suited for this task because radar composites are inherently spatial: precipitation patterns have local structure and translational regularity that convolutions can exploit efficiently. The U-Net combines an encoder branch — which progressively reduces spatial resolution while increasing feature depth to capture large-scale patterns — with a decoder branch that restores spatial resolution using skip connections from the encoder.

---

## Approach

| Component | Choice |
|---|---|
| Architecture | U-Net (encoder–decoder with skip connections) |
| Input | 3 radar frames (log1p-scaled reflectivity), 501×371 px |
| Output | 1 predicted frame at +30 min |
| Loss | Weighted L1 — upweights high-intensity pixels to counter class imbalance |
| Optimizer | AdamW, lr=1e-4, weight decay=1e-4 |
| Regularisation | BatchNorm + Dropout2d (0.1) + early stopping (patience=10) |

The model is trained in log space to stabilize the heavy-tailed precipitation distribution, but weights the loss in physical space to ensure high-intensity events drive the gradients.

The weighted loss is defined as:

```
L = mean( (1 + R) * |pred - target| )
```

where pred and target are in log1p(mm/10min) space, and R = expm1(target) converts the target back to rain rate in mm/10min. This gives dry/light-rain pixels a baseline weight of 1, while progressively increasing the penalty for errors in heavier precipitation.

### Temporal consistency (in progress)
The three decoder branches are connected via cross-connections at 64×48 resolution: +10 min decoder features are injected into the +20 min decoder, and +20 min features into the +30 min decoder. This encourages the model to produce temporally coherent predictions at the mesoscale. Full evaluation via temporal autocorrelation diagnostics is ongoing.

---

## Repository structure

```
├── notebooks
    ├── nowcast_01_preprocessing.ipynb   # raw radar → npz samples
    ├── nowcast_02_enrich_split.ipynb    # temporal train/val/test split + metadata
    └── nowcast_03_train.ipynb           # initial prototype (single-GPU, small data)
├── nowcast_04_train.py          # full training script (SLURM)
├── nowcast_05_test.py               # inference + metrics on held-out test set
├── data/
│   ├── phase2_samples_meta.csv           # event metadata
│   └── phase2_samples_meta_enriched.csv  # metadata with train/val/test split
├── radar_data/                          # raw radar data (not tracked — MeteoSwiss, not public)
├── scripts/
│   ├── run_train.sh                 # SLURM job for training
│   └── run_test.sh                  # SLURM job for evaluation
├── figures/                         # sample output plots 
│ 
└── utils/                           # helpers to be added
```

---

## Results

Evaluation on held-out test set. Best configuration: **weighted L1 loss** with linear rain-rate penalty (`weight = 1 + R`, where R is mm/10min) — heavier rainfall events receive proportionally higher loss weight to counter the class imbalance between rainy and dry pixels. Switching to plain MSE produced blurrier predictions and worse skill scores at moderate-to-high intensities. Persistence = last input frame held constant as the +30 min forecast.

| Metric | U-Net | Persistence |
|---|---|---|
| MAE (mm/10min) | **0.05** | 0.08 |
| RMSE (mm/10min) | **0.33** | 0.49 |
| CSI @ 0.1 mm/10min | **0.64** | 0.47 |
| CSI @ 0.5 mm/10min | **0.41** | 0.24 |
| CSI @ 1.0 mm/10min | **0.29** | 0.13 |
| FSS @ 0.1 mm/10min, 8 px | **0.88** | 0.75 |
| FSS @ 0.1 mm/10min, 32 px | **0.94** | 0.90 |
| FSS @ 0.5 mm/10min, 8 px | **0.59** | 0.52 |
| FSS @ 0.5 mm/10min, 32 px | 0.67 | **0.76** |
| FSS @ 1.0 mm/10min, 8 px | **0.39** | 0.35 |
| FSS @ 1.0 mm/10min, 32 px | 0.47 | **0.61** |

Preliminary results in figures/ show the model captures the overall spatial structure of precipitation fields, but predictions are smoother than observed — fine-scale intensity peaks are underestimated, a common trait of pixel-wise loss functions.

---

## Quickstart

### 1. Install dependencies

```bash
conda env create -f environment.yml
conda activate nowprecip
```

### 2. Prepare data

Place the following files in `radar_data/`:
- `phase2_samples.npz` — input/output radar stacks
- `phase2_samples_meta_enriched.csv` — metadata with `split` column

See `nowcast_01_preprocessing.ipynb` and `nowcast_02_enrich_split.ipynb` for how these are generated.

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

MeteoSwiss radar composite data ('phase2_samples.npz'). Not publicly available — contact MeteoSwiss for access.

The event metadata and train/val/test split (`phase2_samples_meta.csv`, `phase2_samples_meta_enriched.csv`) are included in this repository to allow full reproducibility of the split for anyone with access to the raw data.
