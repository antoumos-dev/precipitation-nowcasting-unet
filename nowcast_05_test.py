import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import uniform_filter

# ============================================================
# CONFIG
# ============================================================
_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "radar_data"
CKPT_DIR  = _HERE / "checkpoints"
OUT_DIR   = _HERE / "test_output"

RUN_NAME     = "unet2_lsd_ltw_rlrop"    # must match training run
BATCH_SIZE   = 8
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD          = (6, 7, 5, 6)
PLOT_INDICES = [1100]            # which test-set samples to visualise

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

class MultiOutputUNet(nn.Module):
    def __init__(self, in_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        f = features
        self.enc1       = EncoderBlock(in_channels, f[0])
        self.enc2       = EncoderBlock(f[0],        f[1])
        self.enc3       = EncoderBlock(f[1],        f[2])
        self.enc4       = EncoderBlock(f[2],        f[3])
        self.bottleneck = DoubleConv(f[3], f[3] * 2)
        for sfx in ('_10', '_20', '_30'):
            setattr(self, f'dec4{sfx}', DecoderBlock(f[3] * 2, f[3]))
            setattr(self, f'dec3{sfx}', DecoderBlock(f[3],     f[2]))
            setattr(self, f'dec2{sfx}', DecoderBlock(f[2],     f[1]))
            setattr(self, f'dec1{sfx}', DecoderBlock(f[1],     f[0]))
            setattr(self, f'final{sfx}', nn.Conv2d(f[0], 1, 1))
        self.cross_10_to_20 = nn.Conv2d(f[3], f[3], 1)
        self.cross_20_to_30 = nn.Conv2d(f[3], f[3], 1)

    def _decode(self, sfx, b, s1, s2, s3, s4, inject=None):
        d4 = getattr(self, f'dec4{sfx}')(b, s4)
        if inject is not None:
            d4 = d4 + inject
        d3 = getattr(self, f'dec3{sfx}')(d4, s3)
        d2 = getattr(self, f'dec2{sfx}')(d3, s2)
        d1 = getattr(self, f'dec1{sfx}')(d2, s1)
        return getattr(self, f'final{sfx}')(d1), d4

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        b     = self.bottleneck(x)
        out10, d4_10 = self._decode('_10', b, s1, s2, s3, s4)
        out20, d4_20 = self._decode('_20', b, s1, s2, s3, s4,
                                    inject=self.cross_10_to_20(d4_10))
        out30, _     = self._decode('_30', b, s1, s2, s3, s4,
                                    inject=self.cross_20_to_30(d4_20))
        out = F.relu(torch.cat([out10, out20, out30], dim=1))
        return out[:, :, 5:506, 6:377]

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
meta_test = meta[test_mask].reset_index(drop=True)
meta_test[["t0_datetime", "event_id", "mean_intensity", "max_intensity", "wet_fraction"]]\
    .rename_axis("idx").to_csv(OUT_DIR / "test_index_lookup.csv")
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
model = MultiOutputUNet(features=[32, 64, 128, 256]).to(DEVICE)
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

preds_log = torch.cat(preds_log).numpy()   # (N, 3, 501, 371)
trues_log = torch.cat(trues_log).numpy()

# Convert to mm/10min
preds_mmh = np.expm1(preds_log)
trues_mmh = np.expm1(trues_log)

# Persistence baseline: last input frame (t0), same for all lead times
persist_mmh = np.expm1(X_test[:, 2:3, :, :])  # (N, 1, 501, 371)

LEAD_NAMES = ["+10min", "+20min", "+30min"]

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

def fss_score(pred, obs, threshold, scale_px):
    pred_bin  = (pred >= threshold).astype(np.float32)
    obs_bin   = (obs  >= threshold).astype(np.float32)
    size      = 2 * scale_px + 1
    pred_frac = uniform_filter(pred_bin, size=size, mode="constant")
    obs_frac  = uniform_filter(obs_bin,  size=size, mode="constant")
    fbs       = np.mean((pred_frac - obs_frac) ** 2)
    fbs_worst = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)
    return 1.0 - fbs / fbs_worst if fbs_worst > 0 else float("nan")

thresholds     = [0.1, 0.2, 0.5, 1.0]
FSS_THRESHOLDS = [0.1, 0.5, 1.0]
FSS_SCALES_PX  = [8, 32]

rows = []
for j, lead in enumerate(LEAD_NAMES):
    pred_j    = preds_mmh[:, j:j+1, :, :]
    true_j    = trues_mmh[:, j:j+1, :, :]

    mae_model  = float(np.mean(np.abs(pred_j   - true_j)))
    rmse_model = float(np.sqrt(np.mean((pred_j - true_j) ** 2)))
    mae_pers   = float(np.mean(np.abs(persist_mmh - true_j)))
    rmse_pers  = float(np.sqrt(np.mean((persist_mmh - true_j) ** 2)))

    csi_model = compute_csi(pred_j,       true_j, thresholds)
    csi_pers  = compute_csi(persist_mmh,  true_j, thresholds)

    print(f"\n--- Lead {lead} ---")
    print(f"  {'Metric':<28} {'U-Net':>8} {'Persist':>8}")
    print("  " + "-" * 46)
    print(f"  {'MAE  (mm/10min)':<28} {mae_model:>8.3f} {mae_pers:>8.3f}")
    print(f"  {'RMSE (mm/10min)':<28} {rmse_model:>8.3f} {rmse_pers:>8.3f}")
    for thr in thresholds:
        print(f"  {'CSI @' + str(thr) + 'mm/10min':<28} {csi_model[thr]:>8.3f} {csi_pers[thr]:>8.3f}")

    for thr in FSS_THRESHOLDS:
        for s in FSS_SCALES_PX:
            fss_m = np.nanmean([fss_score(pred_j[i, 0],      true_j[i, 0],      thr, s) for i in range(len(pred_j))])
            fss_p = np.nanmean([fss_score(persist_mmh[i, 0], true_j[i, 0], thr, s) for i in range(len(pred_j))])
            print(f"  {'FSS @' + str(thr) + 'mm scale=' + str(s) + 'px':<28} {fss_m:>8.3f} {fss_p:>8.3f}")
            rows += [
                {"run": RUN_NAME,      "lead": lead, "metric": "fss", "threshold_mm": thr, "scale_px": s, "value": round(fss_m, 3)},
                {"run": "persistence", "lead": lead, "metric": "fss", "threshold_mm": thr, "scale_px": s, "value": round(fss_p, 3)},
            ]

    rows += [
        {"run": RUN_NAME,      "lead": lead, "metric": "mae",  "threshold_mm": None, "scale_px": None, "value": round(mae_model,  3)},
        {"run": RUN_NAME,      "lead": lead, "metric": "rmse", "threshold_mm": None, "scale_px": None, "value": round(rmse_model, 3)},
        {"run": "persistence", "lead": lead, "metric": "mae",  "threshold_mm": None, "scale_px": None, "value": round(mae_pers,   3)},
        {"run": "persistence", "lead": lead, "metric": "rmse", "threshold_mm": None, "scale_px": None, "value": round(rmse_pers,  3)},
    ] + [
        {"run": RUN_NAME,      "lead": lead, "metric": "csi", "threshold_mm": thr, "scale_px": None, "value": round(csi_model[thr], 3)}
        for thr in [0.1, 0.5, 1.0]
    ] + [
        {"run": "persistence", "lead": lead, "metric": "csi", "threshold_mm": thr, "scale_px": None, "value": round(csi_pers[thr],  3)}
        for thr in [0.1, 0.5, 1.0]
    ]

pd.DataFrame(rows).to_csv(RUN_OUT_DIR / f"metrics_{RUN_NAME}.csv", index=False)

# ============================================================
# PRECIPITATION COLORMAP (radar-style: white → blue → green → yellow → red)
# ============================================================
from matplotlib.colors import BoundaryNorm, ListedColormap

PRECIP_LEVELS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
PRECIP_COLORS = [
    "#c8f0ff",  # 0.1  very light blue
    "#7ec8f5",  # 0.2  light blue
    "#3399cc",  # 0.5  blue
    "#00cc66",  # 1.0  green
    "#66cc00",  # 2.0  lime
    "#ffdd00",  # 3.0  yellow
    "#ff9900",  # 4.0  orange
    "#ff4400",  # 5.0  red-orange
    "#cc0000",  # 7.0  red
    "#800000",  # 10.0 dark red (extend="max" covers >10)
]
cmap_precip = ListedColormap(["#ffffff"] + PRECIP_COLORS)
norm_precip = BoundaryNorm([0.0] + PRECIP_LEVELS + [100.0], cmap_precip.N)

# ============================================================
# PLOT: selected events (2 rows × 3 lead-time columns, one file each)
# ============================================================
ROW_LABEL = ["Observation", "U-Net"]

for plot_num, i in enumerate(PLOT_INDICES, start=1):
    ts      = pd.Timestamp(meta_test.loc[i, "t0_datetime"])
    ts_str  = ts.strftime("%Y%m%d_%H%M")
    ts_nice = ts.strftime("%Y-%m-%d %H:%M UTC")

    fig, axes = plt.subplots(
        2, 3, figsize=(13, 7),
        gridspec_kw={"wspace": 0.04, "hspace": 0.10},
    )

    for r, (row_data, row_label) in enumerate(zip([trues_mmh[i], preds_mmh[i]], ROW_LABEL)):
        for j, lead in enumerate(LEAD_NAMES):
            ax = axes[r, j]
            im = ax.imshow(np.rot90(row_data[j]), norm=norm_precip, cmap=cmap_precip, origin="upper")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(lead, fontsize=12, pad=6)
        axes[r, 0].text(
            -0.06, 0.5, row_label,
            transform=axes[r, 0].transAxes,
            va="center", ha="right", fontsize=12, fontweight="bold", rotation=90,
        )

    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(),
        orientation="vertical", pad=0.02, shrink=0.85, extend="max",
    )
    cbar.set_label("mm / 10 min", fontsize=11)
    cbar.set_ticks(PRECIP_LEVELS)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle(f"Event {plot_num}  |  {ts_nice}  —  {RUN_NAME}", fontsize=12, y=1.01)
    out_path = RUN_OUT_DIR / f"event_{plot_num:02d}_{ts_str}_{RUN_NAME}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")



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

k_ref, _ = radial_power_spectrum(trues_mmh[0, 0])
colors    = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots(figsize=(7, 5))
for j, lead in enumerate(LEAD_NAMES):
    psd_obs  = np.zeros_like(k_ref)
    psd_pred = np.zeros_like(k_ref)
    for i in range(len(trues_mmh)):
        _, p_obs  = radial_power_spectrum(trues_mmh[i, j])
        _, p_pred = radial_power_spectrum(preds_mmh[i, j])
        psd_obs  += p_obs
        psd_pred += p_pred
    psd_obs  /= len(trues_mmh)
    psd_pred /= len(preds_mmh)
    ax.loglog(k_ref, psd_obs,  color=colors[j], linestyle="-",  label=f"Observed {lead}")
    ax.loglog(k_ref, psd_pred, color=colors[j], linestyle="--", label=f"Predicted {lead}")

ax.set_xlabel("Wavenumber (cycles / pixel)")
ax.set_ylabel("Power spectral density")
ax.set_title(f"Radial Power Spectrum — {RUN_NAME}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(RUN_OUT_DIR / f"power_spectrum_{RUN_NAME}.png", dpi=150)
print(f"Power spectrum plot saved to {RUN_OUT_DIR / f'power_spectrum_{RUN_NAME}.png'}")

# ============================================================
# TEMPORAL AUTOCORRELATION
# ============================================================
def spatial_corr(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")

t0_mmh       = np.expm1(X_test[:, 2, :, :])
lead_minutes = [10, 20, 30]
N            = len(t0_mmh)

corr = {
    "Observed":    [np.nanmean([spatial_corr(t0_mmh[i], trues_mmh[i, j])   for i in range(N)]) for j in range(3)],
    "Model":       [np.nanmean([spatial_corr(t0_mmh[i], preds_mmh[i, j])   for i in range(N)]) for j in range(3)],
    "Persistence": [np.nanmean([spatial_corr(t0_mmh[i], persist_mmh[i, 0]) for i in range(N)]) for _ in range(3)],
}

print("\nTemporal Autocorrelation (corr with t0):")
print(f"  {'Lead':<8}" + "".join(f"{k:>14}" for k in corr))
print("  " + "-" * 50)
for j, lt in enumerate(lead_minutes):
    print(f"  {lt:>+4}min  " + "".join(f"{corr[k][j]:>14.4f}" for k in corr))

styles = {"Observed": ("black", "o", "-"), "Model": ("tab:blue", "s", "--"), "Persistence": ("tab:orange", "^", ":")}
fig, ax = plt.subplots(figsize=(6, 4))
for label, (color, marker, ls) in styles.items():
    ax.plot(lead_minutes, corr[label], marker=marker, color=color, linestyle=ls, label=label)
ax.set(xlabel="Lead time (min)", ylabel="Spatial correlation with t0",
       title=f"Temporal Autocorrelation — {RUN_NAME}", ylim=(0, 1.05))
ax.set_xticks(lead_minutes)
ax.legend()
plt.tight_layout()
plt.savefig(RUN_OUT_DIR / f"temporal_autocorr_{RUN_NAME}.png", dpi=150)
print(f"Temporal autocorrelation plot saved to {RUN_OUT_DIR / f'temporal_autocorr_{RUN_NAME}.png'}")
