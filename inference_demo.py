"""
inference_demo.py
=================
Colab-friendly inference helpers for trained water-level models.

This module reuses the training dataset and EfficientNet regressor so image
preprocessing, ROI cropping, and model architecture stay consistent.
"""

import os
import re
import pickle
from datetime import timedelta
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import train_demo as td


SITE_ID = "0204295505"
SITE_NAME = "LITTLE NECK CREEK AT PINEWOOD RD AT VA BEACH, VA"
CAM_ID = "VA_Little_Neck_Creek_at_Pinewood_Road_at_Virginia_Beach"
S3_OVERLAY_BASE = (
    "https://usgs-nims-images.s3.amazonaws.com/overlay/"
    "VA_Little_Neck_Creek_at_Pinewood_Road_at_Virginia_Beach/"
)
USGS_PARAMETER_CODE = "62620"

# PIL/tensor crop format: (x1, y1, x2, y2)
PINEWOOD_ROI = (951, 0, 1136, 1920)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _uploaded_file_path(uploaded_file) -> Optional[str]:
    """Return a filesystem path from a Gradio upload object or raw path string."""
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    return (
        getattr(uploaded_file, "name", None)
        or getattr(uploaded_file, "path", None)
        or getattr(uploaded_file, "orig_name", None)
    )


def collect_image_files(
    image_folder: Optional[str] = None,
    uploaded_files: Optional[Iterable] = None,
) -> list[str]:
    """Collect image paths from a folder and/or Gradio uploaded files."""
    paths: list[str] = []

    if image_folder:
        folder = os.path.expanduser(str(image_folder).strip())
        if folder and os.path.isdir(folder):
            for root, _, files in os.walk(folder):
                for filename in files:
                    if filename.lower().endswith(IMAGE_EXTENSIONS):
                        paths.append(os.path.join(root, filename))
        elif folder:
            raise ValueError(f"Image folder does not exist: {folder}")

    if uploaded_files:
        for uploaded in uploaded_files:
            path = _uploaded_file_path(uploaded)
            if path and path.lower().endswith(IMAGE_EXTENSIONS) and os.path.exists(path):
                paths.append(path)

    deduped = sorted(dict.fromkeys(os.path.abspath(p) for p in paths))
    if not deduped:
        raise ValueError("No test images found. Provide an image folder or upload images.")
    return deduped


def load_scaler(scaler_path: str):
    path = os.path.expanduser(str(scaler_path).strip())
    if not path or not os.path.exists(path):
        raise ValueError(f"Scaler file not found: {scaler_path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _clean_state_dict(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_model(
    model_path: str,
    device,
    fallback_backbone: str = "tf_efficientnet_b3.ns_jft_in1k",
    param_freeze_ratio: float = 0.65,
):
    """Load a training checkpoint or plain model state_dict."""
    path = os.path.expanduser(str(model_path).strip())
    if not path or not os.path.exists(path):
        raise ValueError(f"Model file not found: {model_path}")

    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        backbone = checkpoint.get("backbone", fallback_backbone)
        checkpoint_img_size = checkpoint.get("input_img_size")
    else:
        state_dict = checkpoint
        backbone = fallback_backbone
        checkpoint_img_size = None

    model = td.EfficientNetRegressor(device, backbone, param_freeze_ratio)
    model.load_state_dict(_clean_state_dict(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model, backbone, checkpoint_img_size


def _parse_datetime_text(value) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    parsed = pd.to_datetime(text, errors="coerce", utc=False)
    if pd.notna(parsed):
        return pd.Timestamp(parsed).tz_localize(None) if pd.Timestamp(parsed).tzinfo else pd.Timestamp(parsed)

    patterns = [
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})",
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[-_]?(\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        parts = [int(p) for p in match.groups()]
        if len(parts) == 5:
            parts.append(0)
        try:
            return pd.Timestamp(*parts)
        except ValueError:
            continue
    return None


def _timestamps_from_file(timestamp_path: Optional[str]) -> dict[str, pd.Timestamp]:
    """Read filename -> timestamp mapping from a loose CSV/TXT file if available."""
    if not timestamp_path:
        return {}
    path = os.path.expanduser(str(timestamp_path).strip())
    if not path or not os.path.exists(path):
        return {}

    mappings: dict[str, pd.Timestamp] = {}
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        filename_cols = [
            c for c in df.columns
            if any(token in c.lower() for token in ["file", "image", "path", "name"])
        ]
        time_cols = [
            c for c in df.columns
            if any(token in c.lower() for token in ["time", "date", "datetime", "timestamp"])
        ]
        for _, row in df.iterrows():
            filenames = [str(row[c]) for c in filename_cols if pd.notna(row[c])]
            timestamps = [_parse_datetime_text(row[c]) for c in time_cols if pd.notna(row[c])]
            timestamp = next((ts for ts in timestamps if ts is not None), None)
            if not timestamp:
                timestamp = _parse_datetime_text(" ".join(str(v) for v in row.values))
            for filename in filenames:
                base = os.path.basename(filename)
                if base and timestamp:
                    mappings[base] = timestamp
        if mappings:
            return mappings
    except Exception:
        pass

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            image_match = re.search(r"([A-Za-z0-9_.-]+\.(?:jpg|jpeg|png))", line, re.I)
            timestamp = _parse_datetime_text(line)
            if image_match and timestamp:
                mappings[os.path.basename(image_match.group(1))] = timestamp
    return mappings


def infer_image_timestamps(image_paths: list[str], timestamp_path: Optional[str]) -> dict[str, Optional[pd.Timestamp]]:
    file_map = _timestamps_from_file(timestamp_path)
    timestamps = {}
    for path in image_paths:
        base = os.path.basename(path)
        timestamps[path] = file_map.get(base) or _parse_datetime_text(base)
    return timestamps


def _nearest_usgs_values(
    timestamps: dict[str, Optional[pd.Timestamp]],
    site_id: str = SITE_ID,
    parameter_code: str = USGS_PARAMETER_CODE,
) -> tuple[dict[str, float], str]:
    valid_times = [ts for ts in timestamps.values() if ts is not None]
    if not valid_times:
        return {}, "No image timestamps were found, so USGS true values were skipped."

    try:
        import hydrofunctions as hf
    except Exception:
        return {}, "hydrofunctions is not installed. Predictions were generated without USGS true values."

    start = (min(valid_times) - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (max(valid_times) + timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        nwis = hf.NWIS(site_id, "iv", start, end, parameterCd=parameter_code)
        usgs_df = nwis.df()
    except Exception as e:
        return {}, f"USGS fetch failed through hydrofunctions: {e}. Predictions were generated without true values."

    if usgs_df is None or usgs_df.empty:
        return {}, "USGS returned no rows for the requested image time range."

    numeric = usgs_df.select_dtypes(include=[np.number])
    if numeric.empty:
        return {}, "USGS response did not include a numeric water-level column."

    matching_cols = [c for c in numeric.columns if parameter_code in str(c)]
    value_col = matching_cols[0] if matching_cols else numeric.columns[0]
    series = numeric[value_col].dropna().sort_index()
    if series.empty:
        return {}, "USGS water-level series was empty after removing missing values."

    index = pd.to_datetime(series.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        index = index.tz_convert(None)
    series.index = index
    series = series[series.index.notna()].sort_index()

    nearest: dict[str, float] = {}
    for path, timestamp in timestamps.items():
        if timestamp is None:
            continue
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        pos = series.index.get_indexer([ts], method="nearest")
        if len(pos) and pos[0] >= 0:
            nearest[path] = float(series.iloc[pos[0]])
    return nearest, f"Fetched nearest USGS true values for {len(nearest)} image(s)."


def _crop_corner_columns(roi: Optional[Tuple[int, int, int, int]]) -> dict:
    if roi is None:
        return {
            "cropped_coords_tl": "",
            "cropped_coords_tr": "",
            "cropped_coords_br": "",
            "cropped_coords_bl": "",
        }
    x1, y1, x2, y2 = roi
    return {
        "cropped_coords_tl": f"({x1}, {y1})",
        "cropped_coords_tr": f"({x2}, {y1})",
        "cropped_coords_br": f"({x2}, {y2})",
        "cropped_coords_bl": f"({x1}, {y2})",
    }


def _plot_outputs(df: pd.DataFrame, output_dir: str) -> dict[str, Optional[str]]:
    paths = {"time_series": None, "scatter": None, "error_time": None}
    plot_df = df.dropna(subset=["usgstrue_wl", "mlpredicted_wl_model1"]).copy()
    if plot_df.empty:
        return paths

    plot_df["plot_time"] = pd.to_datetime(plot_df["dt_pdatetime"], errors="coerce")
    plot_df = plot_df.sort_values("plot_time")

    ts_path = os.path.join(output_dir, "inference_timeseries.png")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(plot_df["plot_time"], plot_df["mlpredicted_wl_model1"], marker="o", label="Predicted")
    ax.plot(plot_df["plot_time"], plot_df["usgstrue_wl"], marker="s", label="USGS observed")
    ax.set_title("Predicted vs USGS Observed Water Level")
    ax.set_xlabel("Image time")
    ax.set_ylabel("Water level")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["time_series"] = ts_path

    scatter_path = os.path.join(output_dir, "inference_predicted_vs_true.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(plot_df["usgstrue_wl"], plot_df["mlpredicted_wl_model1"], alpha=0.7)
    lo = float(min(plot_df["usgstrue_wl"].min(), plot_df["mlpredicted_wl_model1"].min()))
    hi = float(max(plot_df["usgstrue_wl"].max(), plot_df["mlpredicted_wl_model1"].max()))
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_title("Predicted vs True")
    ax.set_xlabel("USGS observed")
    ax.set_ylabel("Predicted")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["scatter"] = scatter_path

    error_path = os.path.join(output_dir, "inference_error_over_time.png")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(plot_df["plot_time"], plot_df["dt_abs_error_model1"], marker="o", color="#dc2626")
    ax.set_title("Absolute Error Over Time")
    ax.set_xlabel("Image time")
    ax.set_ylabel("Absolute error")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(error_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["error_time"] = error_path
    return paths


def run_inference(
    model_path: str,
    scaler_path: str,
    image_folder: Optional[str] = None,
    uploaded_files: Optional[Iterable] = None,
    timestamp_file_path: Optional[str] = None,
    input_img_size: int = 600,
    batch_size: int = 1,
    use_pinewood_roi: bool = True,
    fetch_usgs_true: bool = False,
    output_dir: str = "water_level_demo/results/inference",
) -> dict:
    """Run model inference and save CSV/plot outputs."""
    os.makedirs(output_dir, exist_ok=True)

    image_paths = collect_image_files(image_folder, uploaded_files)
    scaler = load_scaler(scaler_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, backbone, checkpoint_img_size = load_model(model_path, device)

    roi = PINEWOOD_ROI if use_pinewood_roi else None
    mappings = {path: 0.0 for path in image_paths}
    ds = td.WaterLevelDataset(
        mappings,
        input_img_size=int(input_img_size),
        roi=roi,
        scaler=scaler,
        training=False,
        include_paths=True,
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        collate_fn=td._collate_fn,
    )

    predictions: list[tuple[str, float]] = []
    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            images, _, paths = batch
            if len(images) == 0:
                continue
            outputs = model(images.float().to(device)).flatten().detach().cpu().numpy()
            values = ds.reverse_scale(outputs)
            predictions.extend((path, float(value)) for path, value in zip(paths, values))

    if not predictions:
        raise ValueError("No valid inference batches were produced. Check that the test images can be opened.")

    timestamps = infer_image_timestamps([p for p, _ in predictions], timestamp_file_path)
    usgs_values = {}
    usgs_message = "USGS true values were not requested."
    if fetch_usgs_true:
        usgs_values, usgs_message = _nearest_usgs_values(timestamps)

    crop_cols = _crop_corner_columns(roi)
    rows = []
    for path, prediction in predictions:
        timestamp = timestamps.get(path)
        true_value = usgs_values.get(path)
        abs_error = abs(prediction - true_value) if true_value is not None else np.nan
        basename = os.path.basename(path)
        rows.append({
            "site_id": SITE_ID,
            "site_name": SITE_NAME,
            "hiviscam_id": CAM_ID,
            "dthivis_image": basename,
            "dfile_path": path,
            "hivis_weblink": S3_OVERLAY_BASE + basename,
            "mlpredicted_wl_model1": prediction,
            "dt_abs_error_model1": abs_error,
            "usgstrue_wl": true_value if true_value is not None else np.nan,
            "dt_pdatetime": timestamp.isoformat(sep=" ") if timestamp is not None else "",
            "dt_tdatetime": timestamp.isoformat(sep=" ") if timestamp is not None else "",
            **crop_cols,
        })

    df = pd.DataFrame(rows, columns=[
        "site_id",
        "site_name",
        "hiviscam_id",
        "dthivis_image",
        "dfile_path",
        "hivis_weblink",
        "mlpredicted_wl_model1",
        "dt_abs_error_model1",
        "usgstrue_wl",
        "dt_pdatetime",
        "dt_tdatetime",
        "cropped_coords_tl",
        "cropped_coords_tr",
        "cropped_coords_br",
        "cropped_coords_bl",
    ])

    csv_path = os.path.join(output_dir, "inference_predictions.csv")
    df.to_csv(csv_path, index=False)
    plot_paths = _plot_outputs(df, output_dir) if df["usgstrue_wl"].notna().any() else {
        "time_series": None,
        "scatter": None,
        "error_time": None,
    }

    status = (
        f"Inference complete for {len(df)} image(s).\n"
        f"Device: {device}\n"
        f"Backbone: {backbone}\n"
        f"Checkpoint image size: {checkpoint_img_size or 'not stored'}\n"
        f"Requested image size: {int(input_img_size)}\n"
        f"ROI: {roi or 'whole image'}\n"
        f"{usgs_message}\n"
        f"CSV saved: {csv_path}"
    )
    return {
        "status": status,
        "dataframe": df,
        "csv_path": csv_path,
        "time_series_plot": plot_paths["time_series"],
        "scatter_plot": plot_paths["scatter"],
        "error_time_plot": plot_paths["error_time"],
    }
