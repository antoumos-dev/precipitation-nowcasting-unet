import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# ============================================================
# CONFIG
# ============================================================
_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "radar_data"
CKPT_DIR  = _HERE / "checkpoints"
OUT_DIR   = _HERE / "test_output"

RUN_NAME   = "wl1_spectral"    # must match training run
BATCH_SIZE = 8
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD        = (6, 7, 5, 6)

RUN_OUT_DIR = OUT_DIR / RUN_NAME
RUN_OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# MODEL (must match training)
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.enc1       = EncoderBlock(in_channels, features[0])
        self.enc2       = EncoderBlock(features[0], features[1])
        self.enc3       = EncoderBlock(features[1], features[2])
        self.enc4       = EncoderBlock(features[2], features[3])
        self.bottleneck = DoubleConv(features[3], features[3] * 2)
        self.dec4       = DecoderBlock(features[3] * 2, features[3])
        self.dec3       = DecoderBlock(features[3], features[2])
        self.dec2       = DecoderBlock(features[2], features[1])
        self.dec1       = DecoderBlock(features[1], features[0])
        self.final      = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        x = self.final(x)
        x = F.relu(x)
        return x[:, :, 5:506, 6:377]

# ============================================================
# LOAD DATA (test split)
# ============================================================
print("Loading data...")
data  = np.load(DATA_DIR / "training_data.npz")
X_all = np.log1p(data["X"]).astype(np.float32)
Y_all = np.log1p(data["Y"]).astype(np.float32)

meta      = pd.read_csv(DATA_DIR / "training_samples_meta_enriched.csv")
test_mask = meta["split"] == "test"
X_test    = X_all[test_mask]
Y_test    = Y_all[test_mask]
print(f"Test samples: {len(X_test)}")

# Pad inputs
X_test_pad = torch.from_numpy(X_test)
X_test_pad = torch.stack([F.pad(x, PAD) for x in X_test_pad])
Y_test_t   = torch.from_numpy(Y_test)

loader = DataLoader(TensorDataset(X_test_pad, Y_test_t),
                    batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=4, pin_memory=True)

# ============================================================
# LOAD MODEL
# ============================================================
model = UNet().to(DEVICE)
ckpt  = torch.load(CKPT_DIR / f"best_model_{RUN_NAME}.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Loaded best model '{RUN_NAME}' from epoch {ckpt['epoch']} (val loss {ckpt['val_loss']:.6f})")

# ============================================================
# INFERENCE
# ============================================================
preds_log, trues_log = [], []

with torch.no_grad():
    for x_b, y_b in loader:
        pred = model(x_b.to(DEVICE)).cpu()
        preds_log.append(pred)
        trues_log.append(y_b)

preds_log = torch.cat(preds_log).numpy()   # (N, 1, 501, 371)
trues_log = torch.cat(trues_log).numpy()

# Convert to mm/10min
preds_mmh = np.expm1(preds_log)
trues_mmh = np.expm1(trues_log)

# Persistence baseline: last input frame (channel index 2) as the +30min prediction
persist_mmh = np.expm1(X_test[:, 2:3, :, :])

# ============================================================
# METRICS
# ============================================================
def compute_csi(pred, obs, thresholds):
    results = {}
    for thr in thresholds:
        tp  = np.sum((pred >= thr) & (obs >= thr))
        fp  = np.sum((pred >= thr) & (obs <  thr))
        fn  = np.sum((pred <  thr) & (obs >= thr))
        results[thr] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    return results

thresholds  = [0.1, 0.2, 0.5, 1.0]

mae_model   = np.mean(np.abs(preds_mmh   - trues_mmh))
rmse_model  = np.sqrt(np.mean((preds_mmh   - trues_mmh) ** 2))
mae_pers    = np.mean(np.abs(persist_mmh - trues_mmh))
rmse_pers   = np.sqrt(np.mean((persist_mmh - trues_mmh) ** 2))

csi_model   = compute_csi(preds_mmh,   trues_mmh, thresholds)
csi_pers    = compute_csi(persist_mmh, trues_mmh, thresholds)

print(f"\n{'Metric':<28} {'U-Net':>8} {'Persist':>8}")
print("-" * 46)
print(f"{'MAE  (mm/10min)':<28} {mae_model:>8.3f} {mae_pers:>8.3f}")
print(f"{'RMSE (mm/10min)':<28} {rmse_model:>8.3f} {rmse_pers:>8.3f}")
for thr in thresholds:
    print(f"{'CSI @' + str(thr) + 'mm/10min':<28} {csi_model[thr]:>8.3f} {csi_pers[thr]:>8.3f}")

# ============================================================
# FSS (Fractions Skill Score)
# ============================================================
def fss_score(pred, obs, threshold, scale_px):
    pred_bin  = (pred >= threshold).astype(np.float32)
    obs_bin   = (obs  >= threshold).astype(np.float32)
    size      = 2 * scale_px + 1
    pred_frac = uniform_filter(pred_bin, size=size, mode="constant")
    obs_frac  = uniform_filter(obs_bin,  size=size, mode="constant")
    fbs       = np.mean((pred_frac - obs_frac) ** 2)
    fbs_worst = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)
    return 1.0 - fbs / fbs_worst if fbs_worst > 0 else float("nan")

FSS_THRESHOLDS = [0.1, 0.5, 1.0]   # mm/10min
FSS_SCALES_PX  = [8, 32]            # neighbourhood radii in pixels

fss_model = {thr: {} for thr in FSS_THRESHOLDS}
fss_pers  = {thr: {} for thr in FSS_THRESHOLDS}
for thr in FSS_THRESHOLDS:
    for s in FSS_SCALES_PX:
        fss_model[thr][s] = np.nanmean([
            fss_score(preds_mmh[i, 0],   trues_mmh[i, 0], thr, s)
            for i in range(len(preds_mmh))
        ])
        fss_pers[thr][s] = np.nanmean([
            fss_score(persist_mmh[i, 0], trues_mmh[i, 0], thr, s)
            for i in range(len(persist_mmh))
        ])

print("\nFSS:")
print(f"  {'Metric':<32} {'U-Net':>8} {'Persist':>8}")
print("  " + "-" * 50)
for thr in FSS_THRESHOLDS:
    for s in FSS_SCALES_PX:
        label = f"FSS @{thr}mm/10min scale={s}px"
        print(f"  {label:<32} {fss_model[thr][s]:>8.3f} {fss_pers[thr][s]:>8.3f}")

rows = (
    [{"run": RUN_NAME,      "metric": "mae",  "threshold_mm": None, "scale_px": None, "value": round(mae_model,  3)},
     {"run": RUN_NAME,      "metric": "rmse", "threshold_mm": None, "scale_px": None, "value": round(rmse_model, 3)},
     {"run": "persistence", "metric": "mae",  "threshold_mm": None, "scale_px": None, "value": round(mae_pers,   3)},
     {"run": "persistence", "metric": "rmse", "threshold_mm": None, "scale_px": None, "value": round(rmse_pers,  3)}]
  + [{"run": RUN_NAME,      "metric": "csi", "threshold_mm": thr, "scale_px": None, "value": round(csi_model[thr], 3)}
     for thr in [0.1, 0.5, 1.0]]
  + [{"run": "persistence", "metric": "csi", "threshold_mm": thr, "scale_px": None, "value": round(csi_pers[thr],  3)}
     for thr in [0.1, 0.5, 1.0]]
  + [{"run": RUN_NAME,      "metric": "fss", "threshold_mm": thr, "scale_px": s, "value": round(fss_model[thr][s], 3)}
     for thr in FSS_THRESHOLDS for s in FSS_SCALES_PX]
  + [{"run": "persistence", "metric": "fss", "threshold_mm": thr, "scale_px": s, "value": round(fss_pers[thr][s],  3)}
     for thr in FSS_THRESHOLDS for s in FSS_SCALES_PX]
)
pd.DataFrame(rows).to_csv(RUN_OUT_DIR / f"metrics_{RUN_NAME}.csv", index=False)

# ============================================================
# PLOT: first 4 test cases
# ============================================================
n_plot = min(4, len(preds_mmh))
vmax   = np.percentile(trues_mmh[:n_plot], 99)

fig, axes = plt.subplots(n_plot, 2, figsize=(8, 3 * n_plot), squeeze=False)
for i in range(n_plot):
    axes[i, 0].imshow(trues_mmh[i, 0], vmin=0, vmax=vmax, cmap="Blues")
    axes[i, 0].set_title(f"Observed [{i}] (mm/10min)")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(preds_mmh[i, 0], vmin=0, vmax=vmax, cmap="Blues")
    axes[i, 1].set_title(f"Predicted [{i}] (mm/10min)")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.savefig(RUN_OUT_DIR / f"test_cases_{RUN_NAME}.png", dpi=150)
print(f"\nPlot saved to {RUN_OUT_DIR / f'test_cases_{RUN_NAME}.png'}")



# ============================================================
# POWER SPECTRA
# ============================================================
def radial_power_spectrum(field):
    """
    Compute the azimuthally-averaged 1-D power spectrum of a 2-D field.
    Returns (wavenumber_bins, mean_power_per_bin).
    """
    ny, nx  = field.shape
    f2d     = np.fft.fft2(field)
    psd2d   = (np.abs(np.fft.fftshift(f2d)) ** 2) / (ny * nx)
    ky      = np.fft.fftshift(np.fft.fftfreq(ny))
    kx      = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY  = np.meshgrid(kx, ky)
    K       = np.sqrt(KX ** 2 + KY ** 2)
    k_bins  = np.linspace(0, K.max(), 64)
    k_mid   = 0.5 * (k_bins[:-1] + k_bins[1:])
    power   = np.array([
        psd2d[(K >= k_bins[i]) & (K < k_bins[i + 1])].mean()
        for i in range(len(k_bins) - 1)
    ])
    return k_mid, power

# average spectra over all test samples
k_ref, _ = radial_power_spectrum(trues_mmh[0, 0])
psd_obs  = np.zeros_like(k_ref)
psd_pred = np.zeros_like(k_ref)
for i in range(len(trues_mmh)):
    _, p_obs  = radial_power_spectrum(trues_mmh[i, 0])
    _, p_pred = radial_power_spectrum(preds_mmh[i, 0])
    psd_obs  += p_obs
    psd_pred += p_pred
psd_obs  /= len(trues_mmh)
psd_pred /= len(preds_mmh)

fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(k_ref, psd_obs,  label="Observed")
ax.loglog(k_ref, psd_pred, label="Predicted", linestyle="--")
ax.set_xlabel("Wavenumber (cycles / pixel)")
ax.set_ylabel("Power spectral density")
ax.set_title(f"Radial Power Spectrum — {RUN_NAME}")
ax.legend()
plt.tight_layout()
plt.savefig(RUN_OUT_DIR / f"power_spectrum_{RUN_NAME}.png", dpi=150)
print(f"Power spectrum plot saved to {RUN_OUT_DIR / f'power_spectrum_{RUN_NAME}.png'}")
