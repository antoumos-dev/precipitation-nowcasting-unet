import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json

# ============================================================
# CONFIG
# ============================================================
_HERE     = Path(__file__).parent
DATA_DIR  = _HERE / "radar_data"
CKPT_DIR  = _HERE / "checkpoints"
LOG_DIR   = _HERE / "logs"
CKPT_DIR.mkdir(exist_ok=True, parents= True)
LOG_DIR.mkdir(exist_ok=True, parents = True)

BATCH_SIZE      = 8
NUM_EPOCHS      = 50
LR              = 1e-4
NUM_WORKERS     = 4
WEIGHT_EXPONENT  = 1    # >1 = more focus on heavy rain, <1 = less
SPECTRAL_WEIGHT  = 0.01  # λ for spectral loss term (tuned: balances L1 ~0.07 and LSD ~9.5 dB)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME    = "unet2_lsd_ltw_rlrop"  # lsd + lead-time weighted + ReduceLROnPlateau

print(f"Device:     {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs:     {NUM_EPOCHS}")
print(f"LR:         {LR}")

print(f"Weighting exponent: {WEIGHT_EXPONENT}")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
data  = np.load(DATA_DIR / "training_data.npz")
X_all = np.log1p(data["X"]).astype(np.float32)
Y_all = np.log1p(data["Y"]).astype(np.float32)
print(f"X_all: {X_all.shape}")
print(f"Y_all: {Y_all.shape}")

# Load enriched metadata with split column
meta = pd.read_csv(DATA_DIR / "training_samples_meta_enriched.csv")
print(f"Meta: {meta.shape}")
print(meta["split"].value_counts())

# ============================================================
# SPLIT
# ============================================================
train_mask = meta["split"] == "train"
val_mask   = meta["split"] == "val"

X_train, Y_train = X_all[train_mask], Y_all[train_mask]
X_val,   Y_val   = X_all[val_mask],   Y_all[val_mask]

print(f"\nX_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")



# ============================================================
# DATASET + DATALOADER
# ============================================================
PAD = (6, 7, 5, 6)  # (left, right, top, bottom) → 371→384, 501→512

class RadarDataset(Dataset):
    def __init__(self, X, Y, augment=False):
        self.X       = torch.from_numpy(X)
        self.Y       = torch.from_numpy(Y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = F.pad(self.X[i], PAD)
        y = self.Y[i]
        if self.augment and torch.rand(1) > 0.5:
            x = torch.flip(x, [-1])
            y = torch.flip(y, [-1])
        return x, y

train_dataset = RadarDataset(X_train, Y_train, augment=True)
val_dataset   = RadarDataset(X_val,   Y_val,   augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train batches: {len(train_loader)}")  # 5984/8 = 748
print(f"Val batches:   {len(val_loader)}")    # 1258/8 = 157

##### Debug: x, y dimensions check ######
x_batch, y_batch = next(iter(train_loader))
print("x:", x_batch.shape)
print("y:", y_batch.shape)

#with torch.no_grad():
#    y_pred = model(x_batch.to(DEVICE))
#print("pred:", y_pred.shape)


# ============================================================
# MODEL
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
    """
    Shared encoder + three parallel decoder branches for +10, +20, +30 min.
    Cross-connections inject +10 dec4 features into the +20 decoder and
    +20 dec4 features into the +30 decoder at 64x48 resolution to enforce
    temporal consistency at the mesoscale.
    """
    def __init__(self, in_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        f = features

        # Shared encoder
        self.enc1       = EncoderBlock(in_channels, f[0])
        self.enc2       = EncoderBlock(f[0],        f[1])
        self.enc3       = EncoderBlock(f[1],        f[2])
        self.enc4       = EncoderBlock(f[2],        f[3])
        self.bottleneck = DoubleConv(f[3], f[3] * 2)

        # Three decoder branches
        for sfx in ('_10', '_20', '_30'):
            setattr(self, f'dec4{sfx}', DecoderBlock(f[3] * 2, f[3]))
            setattr(self, f'dec3{sfx}', DecoderBlock(f[3],     f[2]))
            setattr(self, f'dec2{sfx}', DecoderBlock(f[2],     f[1]))
            setattr(self, f'dec1{sfx}', DecoderBlock(f[1],     f[0]))
            setattr(self, f'final{sfx}', nn.Conv2d(f[0], 1, 1))

        # Cross-connections at 64×48 (dec4 output): 1×1 conv, channel-preserving
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

        out = F.relu(torch.cat([out10, out20, out30], dim=1))  # (B, 3, H, W)
        return out[:, :, 5:506, 6:377]  # crop back to (501, 371)

def weighted_mse(pred, target):
    w = (1.0 + torch.expm1(target)) ** WEIGHT_EXPONENT
    return (w * (pred - target) ** 2).mean()

def weighted_l1(pred, target):
    w = (1.0 + torch.expm1(target)) ** WEIGHT_EXPONENT
    return (w * torch.abs(pred - target)).mean()

def lsd_loss(pred, target):
    # cuFFT rejects these shapes on this driver — run FFT on CPU, return loss to device
    pred_p   = F.pad(pred.contiguous(),   (0, 13, 0, 11)).cpu()  # 371→384, 501→512
    target_p = F.pad(target.contiguous(), (0, 13, 0, 11)).cpu()
    S_pred   = torch.abs(torch.fft.fft2(pred_p))   ** 2
    S_target = torch.abs(torch.fft.fft2(target_p)) ** 2
    log_ratio = 10.0 * torch.log10((S_target + 1e-8) / (S_pred + 1e-8))
    return torch.sqrt((log_ratio ** 2).mean()).to(pred.device)

def gradient_loss(pred, target):
    # Fallback if fft2 is also unsupported — same anti-blur effect, no FFT needed
    pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

LEAD_WEIGHTS = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1)  # +10, +20, +30

def combined_loss(pred, target):
    w = LEAD_WEIGHTS.to(pred.device) * (1.0 + torch.expm1(target)) ** WEIGHT_EXPONENT
    l1 = (w * torch.abs(pred - target)).mean()
    return l1 + SPECTRAL_WEIGHT * lsd_loss(pred, target)

LOSS_FN = combined_loss

model     = MultiOutputUNet(features=[32, 64, 128, 256]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# TRAINING LOOP
# ============================================================

# Track losses for logging
train_losses = []
val_losses   = []

best_val_loss    = float("inf")  # best val loss seen so far
patience         = 10            # stop if no improvement for 10 epochs
epochs_no_improve = 0            # counter

print("\nStarting training...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Best':>6}")
print("-" * 40)

for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    model.train()
    train_loss = 0.0

    for x_b, y_b in train_loader:
        x_b = x_b.to(DEVICE)
        y_b = y_b.to(DEVICE)  # (B, 3, 501, 371) — +10, +20, +30 min

        optimizer.zero_grad()
        pred = model(x_b)
        loss = LOSS_FN(pred, y_b)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(DEVICE)
            y_b = y_b.to(DEVICE)

            pred     = model(x_b)
            val_loss += LOSS_FN(pred, y_b).item()

    val_loss /= len(val_loader)

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {'*' if is_best else ''} lr={current_lr:.2e}")

    # --------------------------------------------------------
    # CHECKPOINTING
    # --------------------------------------------------------
    # Save every 5 epochs
    if epoch % 5 == 0:
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss":      train_loss,
            "val_loss":        val_loss,
        }, CKPT_DIR / f"{RUN_NAME}_checkpoint_epoch_{epoch:03d}.pt")

    # Save best model separately
    if is_best:
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "val_loss":    val_loss,
        }, CKPT_DIR / f"best_model_{RUN_NAME}.pt")
        
    # Save loss log every epoch
    pd.DataFrame({
        "epoch":      list(range(1, epoch + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    }).to_csv(LOG_DIR / f"losses_{RUN_NAME}.csv", index=False)

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------
    if epochs_no_improve >= patience:
        print(f"\nEarly stopping at epoch {epoch} — no improvement for {patience} epochs")
        break

print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


