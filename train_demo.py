"""
train_demo.py
=============
Water Level Prediction Training Module – Google Colab Demo Version
Adapted from the production training pipeline.

Key changes from production:
  - No AWS / S3 / boto3 dependencies
  - Results saved to /content/water_level_demo/results/ or Google Drive
  - Mixed-precision (torch.cuda.amp) support
  - Num_workers capped at 2 for Colab stability
  - CUDA OOM error is caught with a helpful message
  - Smaller EfficientNet option (b3 vs l2) for T4 memory budget
  - ROI cropping preserved from the production Datasets class
  - StandardScaler saved as scaler.pkl (separate file), matching production
"""

import os
import time
import json
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe inside Colab/threads
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
# torch.amp is the unified API (works for PyTorch >= 1.10 and avoids the
# FutureWarning raised by the deprecated torch.cuda.amp module on PT >= 2.4)
import torch.amp as torch_amp
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    ToPILImage, ToTensor, Normalize, Resize,
    ColorJitter, RandomPerspective, Compose,
)
from torchvision.io import read_image

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import timm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalisation (same as production)
MODEL_CONFIG = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# Default Pinewood ROI  (x1, y1, x2, y2)  – XYXY format (vertical gauge strip)
DEFAULT_ROI = {
    "Pinewood Road": (951, 0, 1136, 1920),
}

# Supported EfficientNet variants (production used l2, b3 is lighter for demo)
BACKBONE_OPTIONS = {
    "efficientnet_l2 (production, ~480 MB)": "tf_efficientnet_l2.ns_jft_in1k",
    "efficientnet_b3 (lighter, ~48 MB)":     "tf_efficientnet_b3.ns_jft_in1k",
    "efficientnet_b0 (smallest, ~20 MB)":    "tf_efficientnet_b0.ns_jft_in1k",
}

# ---------------------------------------------------------------------------
# Column auto-detection helpers  (mirrors production detect_usgs_column)
# ---------------------------------------------------------------------------

IMAGE_COL_CANDIDATES  = ["image_path", "dfile_path", "filename", "file", "image"]
TARGET_COL_CANDIDATES = [
    "water_level", "usgstrue_wl", "gage_height", "target", "label",
    "estuary_or_ocean_water_surface_elevation_above_navd_1988",
    "water_surface_elevation", "stream_water_level",
    "reservoir_water_surface_elevation", "elevation",
]
TIME_COL_CANDIDATES   = ["dt_pdatetime", "timestamp", "datetime", "date_time", "dt_image"]


def detect_columns(df: pd.DataFrame):
    """
    Auto-detect image-path, target, and optional timestamp columns.
    Returns (img_col, target_col, time_col) – time_col may be None.
    Raises ValueError with a helpful message if required columns are missing.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(candidates):
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
            # partial match
            for col_lower, col in cols_lower.items():
                if cand.lower() in col_lower:
                    return col
        return None

    img_col    = _find(IMAGE_COL_CANDIDATES)
    target_col = _find(TARGET_COL_CANDIDATES)
    time_col   = _find(TIME_COL_CANDIDATES)

    missing = []
    if img_col is None:
        missing.append(
            f"image-path column (tried: {', '.join(IMAGE_COL_CANDIDATES)})"
        )
    if target_col is None:
        missing.append(
            f"water-level column (tried: {', '.join(TARGET_COL_CANDIDATES)})"
        )

    if missing:
        raise ValueError(
            "❌ Could not detect required CSV columns:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + f"\n\nAvailable columns: {list(df.columns)}"
        )

    return img_col, target_col, time_col


# ---------------------------------------------------------------------------
# Image–label mapping builder
# ---------------------------------------------------------------------------

def build_image_label_mapping(
    df: pd.DataFrame,
    img_col: str,
    target_col: str,
    image_dir: str,
    roi: Optional[Tuple] = None,
    max_images: int = 500,
    seed: int = 42,
) -> dict:
    """
    Build {absolute_image_path: float_water_level} from the CSV.

    Handles two CSV styles:
      1. CSV column contains absolute/relative paths to images.
      2. CSV column contains bare filenames; images are looked up in image_dir.

    ROI is NOT applied here – it is applied inside the Dataset.__getitem__
    at load time (matches production behaviour).
    """
    rng = np.random.default_rng(seed)
    mappings = {}

    for _, row in df.iterrows():
        raw_path = str(row[img_col]).strip()
        target   = row[target_col]

        # Skip NaN / non-finite targets
        try:
            target = float(target)
            if not np.isfinite(target):
                continue
        except (ValueError, TypeError):
            continue

        # Resolve path
        candidate_paths = [
            raw_path,
            os.path.join(image_dir, raw_path),
            os.path.join(image_dir, os.path.basename(raw_path)),
        ]
        resolved = None
        for cp in candidate_paths:
            if os.path.exists(cp) and os.path.getsize(cp) > 0:
                resolved = os.path.abspath(cp)
                break

        if resolved is not None:
            mappings[resolved] = target

    if not mappings:
        raise ValueError(
            "❌ No valid image–label pairs found. "
            "Check that your image paths in the CSV match the uploaded images."
        )

    # Sub-sample if requested
    if len(mappings) > max_images:
        keys   = list(mappings.keys())
        chosen = rng.choice(len(keys), size=max_images, replace=False)
        mappings = {keys[i]: mappings[keys[i]] for i in chosen}

    print(f"✅ Built mapping with {len(mappings)} valid image–label pairs.")
    return mappings


# ---------------------------------------------------------------------------
# ROI preview helper
# ---------------------------------------------------------------------------

def preview_roi(image_path: str, roi: Tuple) -> Tuple:
    """
    Returns (original_pil, cropped_pil) for the given image and ROI.
    roi = (x1, y1, x2, y2)  – XYXY format
    PIL.Image.crop() expects (left, upper, right, lower) = (x1, y1, x2, y2)
    """
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = roi
    cropped = img.crop((x1, y1, x2, y2))
    return img, cropped


# ---------------------------------------------------------------------------
# Dataset  (mirrors production SafeDataset + ROI cropping from Datasets.py)
# ---------------------------------------------------------------------------

class WaterLevelDataset(Dataset):
    """
    PyTorch Dataset for water-level prediction.

    Mirrors the production SafeDataset with:
      • ROI cropping before resize  (matches Datasets.py behaviour)
      • Training-only augmentation (ColorJitter + RandomPerspective)
      • StandardScaler target normalisation
      • Graceful None return on broken images
    """

    def __init__(
        self,
        mappings: dict,
        input_img_size: int,
        roi: Optional[Tuple],
        scaler: Optional[StandardScaler] = None,
        training: bool = True,
    ):
        self.image_paths = list(mappings.keys())
        self.targets_raw = [mappings[p] for p in self.image_paths]
        self.training    = training
        self.roi         = roi           # (x1, y1, x2, y2) XYXY or None

        # ── Target scaling ────────────────────────────────────────────────
        if scaler is None:
            self.scaler = StandardScaler()
            self.targets_scaled = (
                self.scaler
                .fit_transform(np.array(self.targets_raw).reshape(-1, 1))
                .flatten()
            )
        else:
            self.scaler = scaler
            self.targets_scaled = (
                self.scaler
                .transform(np.array(self.targets_raw).reshape(-1, 1))
                .flatten()
            )

        # ── Image transforms ──────────────────────────────────────────────
        t = [ToPILImage(), Resize((input_img_size, input_img_size))]
        if self.training:
            t += [
                ColorJitter(brightness=(0.9, 1.2), contrast=(0.6, 1.4),
                            saturation=(0.6, 1.4), hue=0),
                RandomPerspective(distortion_scale=0.1),
            ]
        t += [ToTensor(), Normalize(mean=MODEL_CONFIG["mean"], std=MODEL_CONFIG["std"])]
        self.transforms = Compose(t)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            path           = self.image_paths[idx]
            target_scaled  = self.targets_scaled[idx]

            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return None

            # Load image as float tensor [C, H, W]
            image = read_image(path).float() / 255.0

            # ROI crop (production Datasets.py crops before resize)
            # roi is (x1, y1, x2, y2) XYXY; tensor is [C, H, W] so crop as [:, y1:y2, x1:x2]
            if self.roi is not None:
                x1, y1, x2, y2 = self.roi
                # Clamp to actual image dimensions
                _, H, W = image.shape
                y1c, y2c = max(0, y1), min(H, y2)
                x1c, x2c = max(0, x1), min(W, x2)
                image = image[:, y1c:y2c, x1c:x2c]

            image = self.transforms(image)
            return image, float(target_scaled)

        except Exception:
            return None

    def reverse_scale(self, arr):
        """Inverse-transform scaled predictions back to original units."""
        return self.scaler.inverse_transform(
            np.asarray(arr).reshape(-1, 1)
        ).flatten()


def _collate_fn(batch):
    """Drop None samples from a batch (handles broken image files)."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ---------------------------------------------------------------------------
# Regression head – defined at module level so state_dict load/save works
# ---------------------------------------------------------------------------

class _RegressionHead(nn.Module):
    """4-layer MLP regression head (mirrors production EfficientNet.RegressionLayers)."""

    def __init__(self, in_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024), nn.GELU(),
            nn.Linear(1024, 512),         nn.GELU(),
            nn.Linear(512, 128),          nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.layers(x.float())


# ---------------------------------------------------------------------------
# EfficientNet regression model
# ---------------------------------------------------------------------------

class EfficientNetRegressor(nn.Module):
    """
    EfficientNet backbone + custom regression head.
    Mirrors the production EfficientNet class exactly.
    param_freeze_ratio controls how much of the backbone is frozen.
    """

    def __init__(self, device, backbone_name: str, param_freeze_ratio: float = 0.7):
        super().__init__()
        print(f"  Loading backbone: {backbone_name} …")
        self.model = timm.create_model(backbone_name, pretrained=True)
        self.model.reset_classifier(0)

        # Freeze first param_freeze_ratio of parameters
        total_params   = sum(p.numel() for p in self.model.parameters())
        freeze_target  = int(total_params * param_freeze_ratio)
        frozen_count   = 0
        for param in self.model.parameters():
            if frozen_count >= freeze_target:
                break
            param.requires_grad = False
            frozen_count += param.numel()

        print(f"  Froze {frozen_count:,} / {total_params:,} backbone parameters.")

        # Probe feature dimension: move model to device first, then run dummy pass
        self.model = self.model.to(device)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            feat  = self.model.forward_features(dummy)
            # forward_features returns (N, C, H, W) for CNNs – use global-pool shape
            n_features = feat.shape[1]

        self.model.classifier = _RegressionHead(n_features)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# Train / validation split helpers
# ---------------------------------------------------------------------------

def split_mappings(
    mappings: dict,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[dict, dict, dict]:
    """
    Random stratified-ish split into train / val / test mappings.
    Mirrors production logic (train_test_split twice).
    """
    paths   = list(mappings.keys())
    targets = [mappings[p] for p in paths]
    n_samples = len(paths)

    if n_samples < 3:
        raise ValueError(
            "At least 3 labelled images are required to create train, validation, "
            f"and test splits. Found {n_samples}."
        )

    if val_ratio <= 0:
        raise ValueError("Validation split must be greater than 0.")
    if test_ratio <= 0:
        raise ValueError("Test split must be greater than 0.")
    if val_ratio + test_ratio >= 1:
        raise ValueError(
            "Validation split plus test split must be less than 1. "
            f"Got val_ratio={val_ratio} and test_ratio={test_ratio}."
        )

    # First split off the test set
    idx_trainval = list(range(len(paths)))
    try:
        idx_trainval, idx_test = train_test_split(
            idx_trainval, test_size=test_ratio, random_state=seed
        )
        # Then split val from trainval
        val_ratio_adj = val_ratio / (1.0 - test_ratio)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_ratio_adj, random_state=seed
        )
    except ValueError as e:
        raise ValueError(
            "Could not create non-empty train, validation, and test splits. "
            "Use more labelled images or reduce the validation/test split ratios. "
            f"Details: {e}"
        ) from e

    if not idx_train or not idx_val or not idx_test:
        raise ValueError(
            "Train, validation, and test splits must each contain at least one image. "
            "Use more labelled images or adjust the split ratios."
        )

    def _make(idxs):
        return {paths[i]: targets[i] for i in idxs}

    return _make(idx_train), _make(idx_val), _make(idx_test)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_model(
    mappings: dict,
    roi: Optional[Tuple],
    results_dir: str,
    # Hyperparameters
    num_epochs: int      = 5,
    batch_size: int      = 4,
    input_img_size: int  = 384,
    learning_rate: float = 2e-4,
    val_ratio: float     = 0.15,
    test_ratio: float    = 0.10,
    param_freeze_ratio: float = 0.7,
    seed: int            = 42,
    backbone_name: str   = "tf_efficientnet_b3.ns_jft_in1k",
    save_to_drive: bool  = False,
    drive_dir: str       = "/content/drive/MyDrive/water_level_demo",
    # Callbacks
    log_callback=None,    # callable(str) – used by Gradio to stream logs
) -> dict:
    """
    Full training pipeline. Returns dict with losses and output paths.

    Follows the production training flow.
    """

    os.makedirs(results_dir, exist_ok=True)

    def _log(msg: str):
        print(msg)
        if log_callback is not None:
            log_callback(msg)

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    _log(f"🖥️  Device: {device}  |  AMP: {use_amp}")

    if device.type == "cuda":
        _log(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        _log(f"   VRAM: {total_mem:.1f} GB")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Data split  (Step 7 equivalent) ──────────────────────────────────
    _log("\n📂 Splitting data …")
    train_map, val_map, test_map = split_mappings(
        mappings, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    _log(f"   Train: {len(train_map)} | Val: {len(val_map)} | Test: {len(test_map)}")

    # Save split summary
    split_summary = pd.DataFrame([
        {"split": "train", "count": len(train_map)},
        {"split": "val",   "count": len(val_map)},
        {"split": "test",  "count": len(test_map)},
    ])
    split_summary_path = os.path.join(results_dir, "split_summary.csv")
    split_summary.to_csv(split_summary_path, index=False)

    # ── Datasets  (Step 8 equivalent) ────────────────────────────────────
    _log("\n🗃️  Building datasets …")
    train_ds = WaterLevelDataset(train_map, input_img_size, roi, scaler=None, training=True)
    val_ds   = WaterLevelDataset(val_map,   input_img_size, roi, scaler=train_ds.scaler, training=False)
    test_ds  = WaterLevelDataset(test_map,  input_img_size, roi, scaler=train_ds.scaler, training=False)

    # Save scaler (production style: separate .pkl file)
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(train_ds.scaler, f)
    _log(f"   Scaler saved → {scaler_path}")

    # Colab-safe DataLoader settings
    num_workers = min(2, os.cpu_count() or 0)
    pin_memory  = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader   = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader  = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=pin_memory,
    )

    # ── Model  (Step 9 equivalent) ────────────────────────────────────────
    _log("\n🏗️  Initialising model …")
    try:
        model = EfficientNetRegressor(device, backbone_name, param_freeze_ratio)
    except Exception as e:
        _log(f"❌ Model init failed: {e}")
        raise

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1, factor=0.5)
    # Use the unified torch.amp API (works on PyTorch >= 1.10, avoids deprecation
    # warnings on >= 2.4).  GradScaler is a no-op when enabled=False (CPU).
    amp_scaler = torch_amp.GradScaler("cuda", enabled=use_amp)

    # ── Training loop  (Step 10 equivalent) ──────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_model_path = os.path.join(results_dir, "best_model.pth")
    start_time      = time.time()

    _log(f"\n🚀 Starting training: {num_epochs} epoch(s) | batch={batch_size} | img={input_img_size}px\n")

    try:
        for epoch in range(num_epochs):
            ep_start = time.time()
            _log(f"{'='*55}")
            _log(f"Epoch {epoch+1}/{num_epochs}")
            _log(f"{'='*55}")

            # ── Train phase ──
            model.train()
            total_train_loss, batch_count = 0.0, 0

            for batch in tqdm(train_loader, desc=f"  Train epoch {epoch+1}", leave=False):
                if batch is None:
                    continue
                images, targets = batch
                if len(images) == 0:
                    continue

                images  = images.to(device)
                targets = targets.to(device).float()

                optimizer.zero_grad()
                with torch_amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images).flatten()
                    loss    = criterion(outputs, targets)

                amp_scaler.scale(loss).backward()
                amp_scaler.step(optimizer)
                amp_scaler.update()

                total_train_loss += loss.item()
                batch_count      += 1

            avg_train_loss = total_train_loss / max(batch_count, 1)
            train_losses.append(avg_train_loss)

            # ── Val phase ──
            model.eval()
            total_val_loss, val_batch_count = 0.0, 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc="  Validation", leave=False):
                    if batch is None:
                        continue
                    images, targets = batch
                    if len(images) == 0:
                        continue
                    images  = images.to(device)
                    targets = targets.to(device).float()
                    with torch_amp.autocast("cuda", enabled=use_amp):
                        outputs = model(images).flatten()
                        loss    = criterion(outputs, targets)
                    total_val_loss   += loss.item()
                    val_batch_count  += 1

            avg_val_loss = total_val_loss / max(val_batch_count, 1)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)
            ep_time = time.time() - ep_start

            _log(
                f"  Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {ep_time:.1f}s"
            )

            # Save best model (production style)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch":               epoch,
                        "model_state_dict":    model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss":          avg_train_loss,
                        "val_loss":            avg_val_loss,
                        "backbone":            backbone_name,
                        "input_img_size":      input_img_size,
                    },
                    best_model_path,
                )
                _log(f"  ⭐ New best model saved! (val_loss={best_val_loss:.4f})")

            if device.type == "cuda":
                torch.cuda.empty_cache()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            msg = (
                "❌ CUDA Out of Memory!\n"
                "   → Reduce batch size or input image size and try again."
            )
            _log(msg)
            raise RuntimeError(msg) from e
        raise

    # ── Save loss history ─────────────────────────────────────────────────
    history_path = os.path.join(results_dir, "training_history.csv")
    pd.DataFrame({
        "epoch":      list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss":   val_losses,
    }).to_csv(history_path, index=False)
    _log(f"\n📄 Training history saved → {history_path}")

    # ── Config snapshot ───────────────────────────────────────────────────
    config = {
        "backbone":          backbone_name,
        "num_epochs":        num_epochs,
        "batch_size":        batch_size,
        "input_img_size":    input_img_size,
        "learning_rate":     learning_rate,
        "val_ratio":         val_ratio,
        "test_ratio":        test_ratio,
        "param_freeze_ratio": param_freeze_ratio,
        "seed":              seed,
        "roi":               list(roi) if roi else None,
        "n_train":           len(train_map),
        "n_val":             len(val_map),
        "n_test":            len(test_map),
        "best_val_loss":     best_val_loss,
        "total_time_s":      round(time.time() - start_time, 1),
        "completed_at":      datetime.now().isoformat(),
    }
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    _log(f"📄 Config saved → {config_path}")

    # ── Training loss plot ────────────────────────────────────────────────
    plot_path = plot_training_loss(train_losses, val_losses, results_dir)

    # ── Optional Google Drive copy ────────────────────────────────────────
    if save_to_drive:
        try:
            import shutil
            os.makedirs(drive_dir, exist_ok=True)
            for fname in ["best_model.pth", "scaler.pkl", "training_history.csv",
                          "training_loss_plot.png", "config.json", "split_summary.csv"]:
                src = os.path.join(results_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(drive_dir, fname))
            _log(f"☁️  Outputs copied to Google Drive: {drive_dir}")
        except Exception as e:
            _log(f"⚠️  Drive copy failed: {e}")

    total_time = time.time() - start_time
    h, rem     = divmod(int(total_time), 3600)
    m, s       = divmod(rem, 60)
    _log(f"\n✅ Training complete in {h}h {m}m {s}s")
    _log(f"   Best val loss: {best_val_loss:.4f}")

    return {
        "train_losses":   train_losses,
        "val_losses":     val_losses,
        "best_val_loss":  best_val_loss,
        "best_model_path": best_model_path,
        "scaler_path":    scaler_path,
        "history_path":   history_path,
        "config_path":    config_path,
        "plot_path":      plot_path,
        "split_summary_path": split_summary_path,
        "total_time_s":   total_time,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_loss(
    train_losses: list,
    val_losses: list,
    results_dir: str,
) -> str:
    """
    Generate and save training/validation loss curve.
    Returns path to saved PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs  = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, marker="o", linewidth=2,
            color="#2563EB", label="Train Loss")
    ax.plot(epochs, val_losses,   marker="s", linewidth=2,
            color="#DC2626", label="Validation Loss", linestyle="--")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(list(epochs))
    fig.tight_layout()

    plot_path = os.path.join(results_dir, "training_loss_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Loss plot saved → {plot_path}")
    return plot_path


# ---------------------------------------------------------------------------
# GPU / environment check (called from Colab Cell 1)
# ---------------------------------------------------------------------------

def check_gpu():
    """Print GPU info and return dict with key stats."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name":    None,
        "total_memory_gb": None,
    }
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["device_name"]     = name
        info["total_memory_gb"] = round(mem, 2)
        print(f"✅ GPU detected: {name}  ({mem:.1f} GB VRAM)")
    else:
        print("⚠️  No GPU detected – training will run on CPU (very slow).")
    return info
