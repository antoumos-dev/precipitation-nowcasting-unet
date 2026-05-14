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
WEIGHT_EXPONENT = 1   # >1 = more focus on heavy rain, <1 = less
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_NAME    = "wl1_linear"    # change per run, e.g. "mse_baseline", "wmse_linear", "wl1_linear"

print(f"Device:     {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs:     {NUM_EPOCHS}")
print(f"LR:         {LR}")

print(f"Weighting exponent: {WEIGHT_EXPONENT}")

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
data  = np.load(DATA_DIR / "phase2_samples.npz")
X_all = np.log1p(data["X"]).astype(np.float32)
Y_all = np.log1p(data["Y"]).astype(np.float32)
print(f"X_all: {X_all.shape}")
print(f"Y_all: {Y_all.shape}")

# Load enriched metadata with split column
meta = pd.read_csv(DATA_DIR / "phase2_samples_meta_enriched.csv")
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

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
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
        x = F.relu(x)                  # non-negative predictions
        return x[:, :, 5:506, 6:377]  # crop back to (501, 371)

def weighted_mse(pred, target):
    w = (1.0 + torch.expm1(target)) ** WEIGHT_EXPONENT
    return (w * (pred - target) ** 2).mean()

def weighted_l1(pred, target):
    # L1 penalises residuals linearly → less regression-to-mean → sharper predictions
    w = (1.0 + torch.expm1(target)) ** WEIGHT_EXPONENT
    return (w * torch.abs(pred - target)).mean()

LOSS_FN = weighted_l1   # swap to weighted_mse to revert

model     = UNet(features=[32, 64, 128, 256]).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

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
        y_b = y_b.to(DEVICE)  # already (8, 1, 501, 371)

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

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>12.6f} {'*' if is_best else ''}")

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


