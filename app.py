"""
app.py – Gradio Training UI for Water Level Prediction Demo
===========================================================
Workflow:
  Tab 1  – Data Acquisition  : pick site, date range, download HiVIS images
                                + auto-fetch USGS sensor data -> labels.csv
  Tab 2  – ROI Settings      : crop region preview
  Tab 3  – Training          : hyperparameters + live log
  Tab 4  – Results           : loss plot + summary
"""

import os
import random
import threading
import queue
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

import train_demo as td
import data_acquisition as da
import inference_demo as inf

# ---------------------------------------------------------------------------
# Paths – auto-detect Colab vs local
# ---------------------------------------------------------------------------
if os.path.exists("/content"):
    BASE_DIR = os.environ.get("WATER_LEVEL_DEMO_BASE_DIR", "/content/water_level_demo")
else:
    BASE_DIR = os.environ.get(
        "WATER_LEVEL_DEMO_BASE_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_level_demo"),
    )

DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(DATA_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
INFERENCE_DATA_DIR = os.path.join(BASE_DIR, "inference", "data")
DEFAULT_LABELS_CSV_PATH = os.path.join(DATA_DIR, "labels.csv")
INFERENCE_OUTPUT_DIR = os.path.join(RESULTS_DIR, "inference")
INFERENCE_ERROR_LOG_PATH = os.path.join(RESULTS_DIR, "inference_error_log.txt")
DRIVE_DIR   = "/content/drive/MyDrive/water_level_demo/results"

for d in [BASE_DIR, DATA_DIR, IMAGES_DIR, RESULTS_DIR, INFERENCE_DATA_DIR, INFERENCE_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Global shared state
# ---------------------------------------------------------------------------
_state = {
    "mappings":   None,   # {abs_image_path: float water_level}
    "roi":        (951, 0, 1136, 1920),   # x1, y1, x2, y2 – vertical gauge strip
    "roi_mode":   "Whole image",
    "csv_path":   None,
}

# ---------------------------------------------------------------------------
# Tab 1 – Data Acquisition handlers
# ---------------------------------------------------------------------------

_acq_queue: queue.Queue = queue.Queue()

def _acquisition_thread(site_name, start_date, end_date, max_images,
                        param_codes, api_key):
    def _log(msg):
        _acq_queue.put(msg)

    try:
        csv_path, matched, roi = da.run_acquisition(
            site_name  = site_name,
            start_date = start_date,
            end_date   = end_date,
            max_images = int(max_images),
            param_codes= param_codes,
            dest_dir   = DATA_DIR,
            api_key    = api_key.strip() if api_key else None,
            log_cb     = _log,
        )

        # Build mappings for training
        df = pd.read_csv(csv_path)
        img_col, target_col, _ = td.detect_columns(df)
        mappings = td.build_image_label_mapping(
            df, img_col, target_col,
            image_dir=IMAGES_DIR,
            roi=None,
            max_images=int(max_images),
            seed=42,
        )
        _state["mappings"] = mappings
        _state["roi"]      = roi
        _state["csv_path"] = csv_path

        _log(f"\nDataset ready: {len(mappings)} labelled images.")
        _log("__DONE_OK__")

    except Exception as e:
        _log(f"\nError: {e}\n{traceback.format_exc()}")
        _log("__DONE_ERR__")


def start_acquisition(site_name, start_date, end_date, max_images,
                      param_selection, api_key):
    """Gradio click handler – streams acquisition log."""

    if not site_name:
        yield "Please select a site.", get_acquisition_summary()
        return

    # Resolve selected parameter codes from checkbox labels
    code_map = {f"{code} – {label}": code
                for code, label in da.USGS_PARAMETERS.items()}
    param_codes = [code_map[s] for s in param_selection if s in code_map]

    if not param_codes:
        yield "Please select at least one USGS parameter.", get_acquisition_summary()
        return

    # Flush queue
    while not _acq_queue.empty():
        _acq_queue.get_nowait()

    t = threading.Thread(
        target=_acquisition_thread,
        args=(site_name, start_date, end_date, max_images,
              param_codes, api_key),
        daemon=True,
    )
    t.start()

    log_lines = []
    done_ok = False
    while True:
        try:
            line = _acq_queue.get(timeout=300)
        except queue.Empty:
            log_lines.append("Timeout – no output for 5 minutes.")
            break
        if line == "__DONE_OK__":
            done_ok = True
            break
        if line == "__DONE_ERR__":
            break
        log_lines.append(line)
        yield "\n".join(log_lines), "Acquisition running..."

    t.join(timeout=10)
    summary = get_acquisition_summary() if done_ok else "Acquisition failed. Dataset summary unchanged."
    yield "\n".join(log_lines), summary


def get_acquisition_summary():
    """Return a short status string about the loaded dataset."""
    m = _state.get("mappings")
    c = _state.get("csv_path")
    if not m:
        return "No dataset loaded yet."
    targets = list(m.values())
    return (
        f"Dataset loaded: {len(m)} images\n"
        f"Water level range: {min(targets):.3f} – {max(targets):.3f} ft\n"
        f"CSV: {c}\n"
        f"ROI: {_state['roi']}"
    )


# ---------------------------------------------------------------------------
# Tab 2 – ROI handlers
# ---------------------------------------------------------------------------

# Pinewood vertical gauge strip – x1, y1, x2, y2 (XYXY format)
PINEWOOD_ROI_XYXY = (951, 0, 1136, 1920)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _format_roi(roi):
    if roi is None:
        return ""
    x1, y1, x2, y2 = [int(v) for v in roi]
    return f"({x1}, {y1}, {x2}, {y2})"


def _available_image_paths():
    mappings = _state.get("mappings")
    if mappings:
        return [
            p for p in mappings.keys()
            if os.path.exists(p) and p.lower().endswith(IMAGE_EXTENSIONS)
        ]
    if not os.path.isdir(IMAGES_DIR):
        return []
    return [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def _image_size(path):
    with Image.open(path) as img:
        return img.size


def _validate_roi(roi, bounds):
    if roi is None:
        raise ValueError("Please select an ROI before starting ROI-based training.")

    try:
        x1, y1, x2, y2 = [int(v) for v in roi]
    except (TypeError, ValueError):
        raise ValueError("ROI coordinates must be integers in the format (x1, y1, x2, y2).")

    if x1 >= x2 or y1 >= y2:
        raise ValueError("ROI coordinates must satisfy x1 < x2 and y1 < y2.")

    width, height = bounds
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(
            "ROI coordinates must be within image boundaries. "
            f"Image size is {width}x{height}; got {_format_roi((x1, y1, x2, y2))}."
        )

    return (x1, y1, x2, y2)


def _preview_path(sample_path=None):
    if sample_path and os.path.exists(sample_path):
        return sample_path
    images = _available_image_paths()
    if not images:
        raise ValueError("No images found. Run Data Acquisition first.")
    return images[0]


def set_roi_mode(mode):
    _state["roi_mode"] = mode
    if mode == "Whole image":
        return (
            gr.update(visible=False),
            "Whole image mode selected. Training will use full original images.",
            None,
            "",
            None,
            [],
        )
    return (
        gr.update(visible=True),
        "ROI cropped image mode selected. Load a sample image and select two corners.",
        None,
        "",
        None,
        [],
    )


def load_random_sample_image():
    images = _available_image_paths()
    if not images:
        return (
            None,
            None,
            None,
            [],
            "",
            None,
            None,
            None,
            None,
            None,
            "No images found. Run Data Acquisition first.",
        )

    img_path = random.choice(images)
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    return (
        img,
        img_path,
        (width, height),
        [],
        "",
        None,
        None,
        None,
        None,
        None,
        f"Loaded sample: {os.path.basename(img_path)} ({width}x{height}). Click two corners to select an ROI.",
    )


def apply_manual_roi(x1, y1, x2, y2, sample_path, sample_size):
    try:
        img_path = _preview_path(sample_path)
        bounds = sample_size or _image_size(img_path)
        roi = _validate_roi((x1, y1, x2, y2), bounds)
        orig, cropped = td.preview_roi(img_path, roi)
        _state["roi"] = roi
        return (
            orig,
            cropped,
            _format_roi(roi),
            f"ROI selected: {_format_roi(roi)}",
            roi,
            [],
        )
    except Exception as e:
        return None, None, "", f"ROI preview failed: {e}", None, []


def select_roi_point(sample_path, sample_size, clicks, current_roi, evt: gr.SelectData):
    try:
        if not sample_path or not os.path.exists(sample_path):
            raise ValueError("Load a random sample image before selecting an ROI.")
        if not sample_size:
            sample_size = _image_size(sample_path)

        point = evt.index
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError("Could not read the selected image point.")

        x, y = int(point[0]), int(point[1])
        width, height = sample_size
        if x < 0 or y < 0 or x >= width or y >= height:
            raise ValueError(
                f"Selected point ({x}, {y}) is outside image boundaries {width}x{height}."
            )

        clicks = [] if not clicks or len(clicks) >= 2 else list(clicks)
        clicks.append((x, y))

        if len(clicks) == 1:
            return (
                gr.update(),
                None,
                _format_roi(current_roi),
                f"First corner selected at ({x}, {y}). Click the opposite corner.",
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                current_roi,
                clicks,
            )

        (x_a, y_a), (x_b, y_b) = clicks
        roi = _validate_roi(
            (min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)),
            sample_size,
        )
        orig, cropped = td.preview_roi(sample_path, roi)
        _state["roi"] = roi
        return (
            orig,
            cropped,
            _format_roi(roi),
            f"ROI selected: {_format_roi(roi)}",
            roi[0],
            roi[1],
            roi[2],
            roi[3],
            roi,
            clicks,
        )
    except Exception as e:
        return (
            gr.update(),
            None,
            _format_roi(current_roi),
            f"ROI selection failed: {e}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            current_roi,
            clicks or [],
        )


def preview_roi_handler(x1, y1, x2, y2):
    # Store ROI in x1, y1, x2, y2 format
    roi = (int(x1), int(y1), int(x2), int(y2))
    _state["roi"] = roi

    mappings = _state.get("mappings")
    if mappings:
        img_path = list(mappings.keys())[0]
    else:
        all_imgs = [
            os.path.join(IMAGES_DIR, f)
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not all_imgs:
            return None, None, "No images found. Run Data Acquisition first."
        img_path = all_imgs[0]

    try:
        orig, cropped = td.preview_roi(img_path, roi)
        status = (
            f"Original: {orig.width}x{orig.height} px | "
            f"Cropped: {cropped.width}x{cropped.height} px | "
            f"ROI: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}"
        )
        return orig, cropped, status
    except Exception as e:
        return None, None, f"Preview failed: {e}"


def site_roi_autofill(site_name):
    info = da.SITE_CATALOG.get(site_name)

    if info:
        # Use Pinewood vertical strip ROI
        if "Pinewood" in site_name or "Little Neck Creek" in site_name:
            x1, y1, x2, y2 = PINEWOOD_ROI_XYXY
        else:
            x1, y1, x2, y2 = info["roi"]

        _state["roi"] = (x1, y1, x2, y2)

        default_sel = _default_param_selection(site_name)

        return x1, y1, x2, y2, f"ROI auto-filled for {site_name}", default_sel

    return (
        951, 0, 1136, 1920,
        "Default Pinewood vertical strip ROI loaded.",
        _default_param_selection(SITE_CHOICES[0]),
    )


# ---------------------------------------------------------------------------
# Tab 3 – Training handlers
# ---------------------------------------------------------------------------

_log_queue: queue.Queue = queue.Queue()
_train_result: dict = {}


def _training_thread(kwargs):
    def _log(msg):
        _log_queue.put(msg)
    try:
        result = td.train_model(**kwargs, log_callback=_log)
        _train_result.update(result)
        _log_queue.put("__DONE__")
    except Exception as e:
        _log_queue.put(f"Training error: {e}")
        _log_queue.put("__DONE__")


def training_config_mode_changed(mode):
    best = td.BEST_TRAINING_CONFIG
    if mode == "Best configuration for training":
        return (
            gr.update(visible=False),
            td.BEST_TRAINING_SUMMARY,
            best["num_epochs"],
            best["batch_size"],
            best["input_img_size"],
            best["learning_rate"],
            best["val_ratio"],
            best["test_ratio"],
            best["param_freeze_ratio"],
            best["random_state"],
            True,
        )
    return (
        gr.update(visible=True),
        "Manual setup selected. Use the controls below to choose custom hyperparameters.",
        5,
        4,
        384,
        2e-4,
        0.15,
        0.10,
        0.7,
        42,
        True,
    )


def _validate_manual_training_values(
    num_images, n_epochs, batch_size, img_size,
    learning_rate, val_ratio, test_ratio,
):
    if int(num_images) <= 0:
        raise ValueError("Number of images must be greater than 0.")
    if int(img_size) <= 0:
        raise ValueError("Image size must be greater than 0.")
    if int(batch_size) <= 0:
        raise ValueError("Batch size must be greater than 0.")
    if float(learning_rate) <= 0:
        raise ValueError("Learning rate must be greater than 0.")
    if int(n_epochs) <= 0:
        raise ValueError("Epochs must be greater than 0.")
    if float(val_ratio) <= 0:
        raise ValueError("Validation split must be greater than 0.")
    if float(test_ratio) <= 0:
        raise ValueError("Test split must be greater than 0.")
    if float(val_ratio) + float(test_ratio) >= 1:
        raise ValueError("Validation split plus test split must be less than 1.")


def start_training(
    training_config_mode,
    site_name,
    num_images, n_epochs, batch_size, img_size,
    learning_rate, val_ratio, test_ratio,
    freeze_ratio, seed,
    use_small_backbone, save_to_drive,
    roi_mode, selected_roi, sample_size,
):
    mappings = _state.get("mappings")
    if not mappings:
        yield "No dataset loaded. Please run Data Acquisition first.", None, None, ""
        return

    use_best_config = training_config_mode == "Best configuration for training"
    config_mode = "best" if use_best_config else "manual"

    if not use_best_config:
        try:
            _validate_manual_training_values(
                num_images, n_epochs, batch_size, img_size,
                learning_rate, val_ratio, test_ratio,
            )
        except ValueError as e:
            yield str(e), None, None, ""
            return

    if roi_mode == "Whole image":
        roi = None
    else:
        try:
            bounds = sample_size
            if not bounds:
                first_image = next(iter(mappings.keys()))
                bounds = _image_size(first_image)
            roi = _validate_roi(selected_roi, bounds)
        except ValueError as e:
            yield str(e), None, None, ""
            return

    n = len(mappings) if use_best_config else min(int(num_images), len(mappings))
    keys = list(mappings.keys())
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(keys), size=n, replace=False)
    sub = {keys[i]: mappings[keys[i]] for i in chosen}

    if roi is not None:
        try:
            for img_path in sub.keys():
                _validate_roi(roi, _image_size(img_path))
        except ValueError as e:
            yield f"Selected ROI is not valid for all training images. {e}", None, None, ""
            return

    backbone = ("tf_efficientnet_b3.ns_jft_in1k" if use_small_backbone
                else "tf_efficientnet_l2.ns_jft_in1k")

    kwargs = dict(
        mappings           = sub,
        roi                = roi,
        results_dir        = RESULTS_DIR,
        num_epochs         = int(n_epochs),
        batch_size         = int(batch_size),
        input_img_size     = int(img_size),
        learning_rate      = float(learning_rate),
        val_ratio          = float(val_ratio),
        test_ratio         = float(test_ratio),
        param_freeze_ratio = float(freeze_ratio),
        seed               = int(seed),
        backbone_name      = backbone,
        config_mode        = config_mode,
        site_name          = site_name,
        save_to_drive      = bool(save_to_drive),
        drive_dir          = DRIVE_DIR,
    )

    while not _log_queue.empty():
        _log_queue.get_nowait()
    _train_result.clear()

    thread = threading.Thread(target=_training_thread, args=(kwargs,), daemon=True)
    thread.start()

    log_lines = []
    while True:
        try:
            line = _log_queue.get(timeout=120)
        except queue.Empty:
            log_lines.append("Timeout – no output for 2 minutes.")
            break
        if line == "__DONE__":
            break
        log_lines.append(line)
        yield "\n".join(log_lines), None, None, "Training is running. Results will appear here when it finishes."

    thread.join(timeout=5)
    loss_img, pred_img, summary = get_training_outputs()
    yield "\n".join(log_lines), loss_img, pred_img, summary


def get_training_outputs():
    if not _train_result:
        return None, None, "No results yet. Run training first."

    plot_path = (
        _train_result.get("loss_plot_path")
        or _train_result.get("loss_curves_site_path")
        or _train_result.get("plot_path")
    )
    loss_img = Image.open(plot_path) if plot_path and os.path.exists(plot_path) else None
    pred_path = _train_result.get("predictions_plot_path")
    pred_img = Image.open(pred_path) if pred_path and os.path.exists(pred_path) else None
    metrics_summary = _train_result.get("metrics_summary", "N/A")

    t = _train_result.get("total_time_s", 0)
    h, rem = divmod(int(t), 3600)
    m, s   = divmod(rem, 60)
    bvl    = _train_result.get("best_val_loss")
    bvl_s  = f"{bvl:.4f}" if isinstance(bvl, (int, float)) else "N/A"

    summary = (
        f"Training Summary\n{'─'*40}\n"
        f"Best validation loss : {bvl_s}\n"
        f"Training time        : {h}h {m}m {s}s\n"
        f"Best model           : {_train_result.get('best_model_path', 'N/A')}\n"
        f"Site model           : {_train_result.get('best_model_site_path', 'N/A')}\n"
        f"Scaler               : {_train_result.get('scaler_path', 'N/A')}\n"
        f"Test results         : {_train_result.get('test_results_csv_path') or _train_result.get('test_results_path', 'N/A')}\n"
        f"Predictions plot     : {_train_result.get('predictions_plot_path') or 'Not generated'}\n"
        f"Test metrics         : {metrics_summary}\n"
        f"Loss plot            : {plot_path or 'N/A'}\n"
        f"Site loss curves     : {_train_result.get('loss_curves_site_path', 'N/A')}\n"
        f"Config               : {_train_result.get('config_path', 'N/A')}\n"
    )
    return loss_img, pred_img, summary


# ---------------------------------------------------------------------------
# Tab 5 – Inference handlers
# ---------------------------------------------------------------------------

def _format_inference_error(error, step, log_lines):
    text = str(error)
    lower = text.lower()
    suggestions = []

    if "model file not found" in lower:
        suggestions.append("Run training first, or set Model .pth path to an existing best_model.pth file.")
    if "scaler file not found" in lower:
        suggestions.append("Run training first, or set Scaler .pkl path to the scaler.pkl saved with this model.")
    if "no hivis camera found" in lower:
        suggestions.append("Choose a site with an available USGS HiVIS camera.")
    if "no images found" in lower or "no images downloaded" in lower:
        suggestions.append("Try a wider date range or a different site; there may be no HiVIS images for that period.")
    if "no usable water-level target" in lower or "water-level parameter" in lower:
        suggestions.append("Select at least one water-level parameter, usually 62620 for tidal sites or 00065 for gage-height sites.")
    if "no images matched" in lower or "within 15 minutes" in lower:
        suggestions.append("The downloaded image times did not align with USGS readings. Try a different date range or parameter.")
    if "some images referenced" in lower:
        suggestions.append("The labels CSV points to missing images. Re-run inference download for the same site/date range.")
    if "no valid inference batches" in lower:
        suggestions.append("Check that downloaded images are valid JPG/PNG files and that the ROI is inside the image bounds.")
    if "size mismatch" in lower or "shape mismatch" in lower:
        suggestions.append("The selected model and scaler/config may not belong to the same training run.")
    if "could not resolve host" in lower or "connection" in lower or "timeout" in lower:
        suggestions.append("Check network access to USGS/GitHub resources and try again.")

    if not suggestions:
        suggestions.append("Check the selected site, date range, parameters, model path, scaler path, and config path.")

    completed_log = "\n".join(log_lines).strip()
    log_block = f"\n\nCompleted log before failure:\n{completed_log}" if completed_log else ""
    suggestion_block = "\n".join(f"- {s}" for s in dict.fromkeys(suggestions))

    return (
        f"Inference stopped during: {step}\n\n"
        f"Reason:\n{text}\n\n"
        f"What to try:\n{suggestion_block}"
        f"{log_block}"
    )


def _format_inference_traceback_error(error, step, log_lines, tb, log_path):
    return (
        _format_inference_error(error, step, log_lines)
        + "\n\nFull Python traceback:\n"
        + tb
        + f"\nFull traceback saved to:\n{log_path}"
    )


def _optional_int(value):
    if value in (None, ""):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def run_inference_handler(
    model_path,
    scaler_path,
    config_path,
    labels_csv_path,
    site_name,
    start_date,
    end_date,
    max_images,
    param_selection,
    api_key,
    input_img_size_override,
    batch_size,
):
    log_lines = []
    current_step = "validating inputs"
    try:
        if not site_name:
            return "Inference failed: Please select a USGS site.", None, None, None, None, None

        code_map = {f"{code} – {label}": code
                    for code, label in da.USGS_PARAMETERS.items()}
        param_codes = [code_map[s] for s in (param_selection or []) if s in code_map]
        if not param_codes:
            return "Inference failed: Please select at least one USGS parameter.", None, None, None, None, None
        if not any(code in da.WATER_LEVEL_TARGET_CODES for code in param_codes):
            targets = ", ".join(da.WATER_LEVEL_TARGET_CODES)
            return (
                "Inference failed: Select at least one water-level parameter "
                f"({targets}) so predictions can be matched against USGS values.",
                None, None, None, None, None,
            )

        def _log(msg):
            log_lines.append(str(msg))

        model_check_path = os.path.expanduser(str(model_path or "").strip())
        scaler_check_path = os.path.expanduser(str(scaler_path or "").strip())
        config_check_path = os.path.expanduser(str(config_path or "").strip())
        labels_check_path = os.path.expanduser(str(labels_csv_path or "").strip())
        _log(f"[inference] Model file exists: {os.path.exists(model_check_path)} | {model_check_path}")
        _log(f"[inference] Scaler file exists: {os.path.exists(scaler_check_path)} | {scaler_check_path}")
        _log(f"[inference] Config file exists: {os.path.exists(config_check_path)} | {config_check_path}")
        _log(f"[inference] Labels CSV exists: {os.path.exists(labels_check_path)} | {labels_check_path}")
        _log(f"[inference] Output directory exists or created: {os.path.isdir(INFERENCE_OUTPUT_DIR)} | {INFERENCE_OUTPUT_DIR}")

        if labels_check_path:
            csv_path = labels_check_path
            current_step = "checking existing labels CSV"
            if not os.path.exists(csv_path):
                raise ValueError(f"Labels CSV not found: {csv_path}")
            matched = len(pd.read_csv(csv_path))
            _log(f"[inference] Using existing labels CSV: {csv_path}")
        else:
            current_step = "downloading HiVIS images and matching USGS parameters"
            csv_path, matched, _ = da.run_acquisition(
                site_name=site_name,
                start_date=start_date,
                end_date=end_date,
                max_images=int(max_images),
                param_codes=param_codes,
                dest_dir=INFERENCE_DATA_DIR,
                api_key=api_key.strip() if api_key else None,
                log_cb=_log,
            )

        site_info = da.SITE_CATALOG[site_name]
        current_step = "resolving the training ROI from config.json"
        roi, roi_message = inf.resolve_roi_from_training_config(
            config_path,
            site_info.get("roi"),
        )
        _log(f"[inference] {roi_message}")

        current_step = "loading model/scaler and running predictions"
        result = inf.run_inference_from_labels(
            labels_csv_path=csv_path,
            input_img_size=_optional_int(input_img_size_override),
            batch_size=int(batch_size),
            model_path=model_path,
            scaler_path=scaler_path,
            site_name=site_name,
            site_info=site_info,
            roi=roi,
            config_path=config_path,
            output_dir=INFERENCE_OUTPUT_DIR,
            log_callback=_log,
        )

        def _open_plot(path):
            return Image.open(path) if path and os.path.exists(path) else None

        status = (
            "\n".join(log_lines)
            + "\n\n"
            + f"Matched inference samples: {matched}\n"
            + roi_message
            + "\n\n"
            + result["status"]
        )

        return (
            status,
            result["dataframe"],
            result["csv_path"],
            _open_plot(result.get("time_series_plot")),
            _open_plot(result.get("scatter_plot")),
            _open_plot(result.get("error_time_plot")),
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        log_path = inf.save_error_traceback(tb, INFERENCE_ERROR_LOG_PATH)
        return (
            _format_inference_traceback_error(e, current_step, log_lines, tb, log_path),
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Inter', sans-serif; }
.section-header {
    font-size: 1.1em; font-weight: 700; color: #1e40af;
    border-bottom: 2px solid #bfdbfe; padding-bottom: 4px; margin-bottom: 8px;
}
.status-box { font-family: monospace; font-size: 0.85em; }
.log-box    { font-family: monospace; font-size: 0.82em; background: #f1f5f9;
              color: #1e293b; border-radius: 6px; padding: 8px; }
.roi-preview img {
    object-fit: contain !important;
    max-height: 420px !important;
}
"""

PARAM_CHOICES = [f"{code} – {label}" for code, label in da.USGS_PARAMETERS.items()]
SITE_CHOICES  = list(da.SITE_CATALOG.keys())


def _param_labels_for_codes(codes):
    return [
        f"{code} – {da.USGS_PARAMETERS.get(code, code)}"
        for code in codes
        if f"{code} – {da.USGS_PARAMETERS.get(code, code)}" in PARAM_CHOICES
    ]


def _default_param_selection(site_name):
    site_info = da.SITE_CATALOG.get(site_name, {})
    return _param_labels_for_codes(site_info.get("default_params", []))


def launch_gradio(share: bool = True, debug: bool = True, show_error: bool = True):
    """Build and launch the Gradio app."""

    with gr.Blocks(
        title="Water Level Prediction – Training Demo",
        css=CSS,
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="gray",
        ),
    ) as demo:

        gr.Markdown(
            """
            # USGS Water Level Prediction – Training Demo
            ### HiVIS Image Download · USGS Sensor Data · EfficientNet Regression
            ---
            Follow the tabs in order: **Acquire Data → Set ROI → Train → Results**.
            Use **Inference** to test a trained model on new images.
            """
        )

        # ===================================================================
        # TAB 1 – Data Acquisition
        # ===================================================================
        with gr.Tab("1 – Acquire Data"):
            gr.Markdown("### Step 1 – Download Images & Sensor Data",
                        elem_classes="section-header")

            with gr.Row():
                with gr.Column(scale=2):
                    site_dd = gr.Dropdown(
                        choices=SITE_CHOICES,
                        label="USGS Site",
                        value=SITE_CHOICES[0],
                    )
                    with gr.Row():
                        start_date_in = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value="2025-01-01",
                            placeholder="2025-01-01",
                        )
                        end_date_in = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            value="2025-03-31",
                            placeholder="2025-03-31",
                        )
                    max_img_slider = gr.Slider(
                        minimum=50, maximum=500, value=200, step=10,
                        label="Max images to download",
                    )

                with gr.Column(scale=2):
                    param_checks = gr.CheckboxGroup(
                        choices=PARAM_CHOICES,
                        value=_default_param_selection(SITE_CHOICES[0]),
                        label="USGS Parameters to include in CSV",
                    )
                    api_key_in = gr.Textbox(
                        label="NIMS API Key (optional – leave blank to try without)",
                        placeholder="Paste your api.waterdata.usgs.gov key here",
                        type="password",
                    )

            acq_btn = gr.Button("Download Data from HiVIS + USGS",
                                variant="primary", size="lg")

            gr.Markdown("### Acquisition Log")
            acq_log = gr.Textbox(
                label="Live progress",
                lines=18,
                interactive=False,
                elem_classes="log-box",
            )

            acq_summary = gr.Textbox(
                label="Dataset Summary",
                lines=5,
                interactive=False,
                elem_classes="status-box",
            )

            acq_btn.click(
                fn=start_acquisition,
                inputs=[site_dd, start_date_in, end_date_in,
                        max_img_slider, param_checks, api_key_in],
                outputs=[acq_log, acq_summary],
            )

        # ===================================================================
        # TAB 2 – ROI Settings
        # ===================================================================
        with gr.Tab("2 – ROI"):
            gr.Markdown("### Step 2 – Region of Interest",
                        elem_classes="section-header")

            roi_mode = gr.Radio(
                choices=["Whole image", "ROI cropped image"],
                value="Whole image",
                label="Training Image Mode",
            )
            roi_mode_status = gr.Textbox(
                label="Mode Status",
                value="Whole image mode selected. Training will use full original images.",
                interactive=False,
            )
            selected_roi_state = gr.State(None)
            roi_clicks_state = gr.State([])
            roi_sample_path_state = gr.State(None)
            roi_sample_size_state = gr.State(None)

            with gr.Group(visible=False) as roi_controls:
                gr.Markdown(
                    "Load a random sample image, click two opposite corners to select "
                    "a rectangular ROI, or use the manual coordinate fields."
                )
                load_sample_btn = gr.Button("Load Random Sample Image")

                roi_sample_img = gr.Image(
                    label="Sample Image",
                    type="pil",
                    interactive=False,
                    height=420,
                    elem_classes="roi-preview",
                )

                roi_coord_text = gr.Textbox(
                    label="Selected ROI Coordinates",
                    placeholder="(x1, y1, x2, y2)",
                    interactive=False,
                )

                with gr.Row():
                    roi_x1 = gr.Number(value=951,  label="x1 (left)")
                    roi_y1 = gr.Number(value=0,    label="y1 (top)")
                    roi_x2 = gr.Number(value=1136, label="x2 (right)")
                    roi_y2 = gr.Number(value=1920, label="y2 (bottom)")

                preview_btn = gr.Button("Preview/Apply Manual ROI")
                cropped_img = gr.Image(
                    label="Crop Preview",
                    type="pil",
                    height=360,
                    elem_classes="roi-preview",
                )
                roi_status = gr.Textbox(label="ROI Status", interactive=False)

            roi_mode.change(
                fn=set_roi_mode,
                inputs=roi_mode,
                outputs=[
                    roi_controls, roi_mode_status, selected_roi_state,
                    roi_coord_text, cropped_img, roi_clicks_state,
                ],
            )

            load_sample_btn.click(
                fn=load_random_sample_image,
                inputs=[],
                outputs=[
                    roi_sample_img, roi_sample_path_state, roi_sample_size_state,
                    roi_clicks_state, roi_coord_text, cropped_img,
                    roi_x1, roi_y1, roi_x2, roi_y2, roi_status,
                ],
            )

            # Auto-fill ROI when site changes in Tab 1
            site_dd.change(
                fn=site_roi_autofill,
                inputs=site_dd,
                outputs=[roi_x1, roi_y1, roi_x2, roi_y2, roi_status, param_checks],
            )
            preview_btn.click(
                fn=apply_manual_roi,
                inputs=[
                    roi_x1, roi_y1, roi_x2, roi_y2,
                    roi_sample_path_state, roi_sample_size_state,
                ],
                outputs=[
                    roi_sample_img, cropped_img, roi_coord_text,
                    roi_status, selected_roi_state, roi_clicks_state,
                ],
            )
            roi_sample_img.select(
                fn=select_roi_point,
                inputs=[
                    roi_sample_path_state, roi_sample_size_state,
                    roi_clicks_state, selected_roi_state,
                ],
                outputs=[
                    roi_sample_img, cropped_img, roi_coord_text, roi_status,
                    roi_x1, roi_y1, roi_x2, roi_y2,
                    selected_roi_state, roi_clicks_state,
                ],
            )

        # ===================================================================
        # TAB 3 – Training
        # ===================================================================
        with gr.Tab("3 – Train"):
            gr.Markdown("### Step 3 – Configure and Start Training",
                        elem_classes="section-header")

            best_cfg = td.BEST_TRAINING_CONFIG
            training_config_mode = gr.Radio(
                choices=["Best configuration for training", "Manual setup"],
                value="Best configuration for training",
                label="Training Configuration Mode",
            )
            training_config_summary = gr.Textbox(
                label="Training Configuration Summary",
                value=td.BEST_TRAINING_SUMMARY,
                interactive=False,
                lines=3,
            )

            with gr.Group(visible=False) as manual_training_controls:
                with gr.Row():
                    with gr.Column():
                        t_num_images = gr.Slider(50, 500, value=200, step=10,
                                                label="Number of images to train on")
                        t_epochs     = gr.Slider(1, 20, value=best_cfg["num_epochs"], step=1,
                                                label="Epochs")
                        t_batch      = gr.Dropdown(choices=[2, 4, 8, 16],
                                                  value=best_cfg["batch_size"],
                                                  label="Batch size")
                        t_img_size   = gr.Dropdown(choices=[224, 384, 512, 600],
                                                  value=best_cfg["input_img_size"],
                                                  label="Input image size (px)")
                    with gr.Column():
                        t_lr         = gr.Number(value=best_cfg["learning_rate"],
                                                label="Learning rate", precision=6)
                        t_val_ratio  = gr.Number(value=best_cfg["val_ratio"],
                                                label="Validation split", precision=2)
                        t_test_ratio = gr.Number(value=best_cfg["test_ratio"],
                                                label="Test split", precision=2)
                        t_freeze     = gr.Slider(0.0, 1.0,
                                                value=best_cfg["param_freeze_ratio"],
                                                step=0.05,
                                                label="Backbone freeze ratio")
                        t_seed       = gr.Number(value=best_cfg["random_state"],
                                                label="Random seed", precision=0)
                    with gr.Column():
                        t_small_model = gr.Checkbox(
                            value=True,
                            label="Use EfficientNet-B3 (recommended – fits on T4 GPU)",
                        )
                        gr.Markdown(
                            """
                            **GPU Memory tips (T4 – 15 GB):**
                            - Best configuration uses EfficientNet-B3 at 384px, batch 8
                            - If out-of-memory: best mode retries batch 4 automatically
                            """
                        )

            t_save_drive  = gr.Checkbox(
                value=False,
                label="Copy results to Google Drive",
            )

            training_config_mode.change(
                fn=training_config_mode_changed,
                inputs=training_config_mode,
                outputs=[
                    manual_training_controls, training_config_summary,
                    t_epochs, t_batch, t_img_size, t_lr, t_val_ratio,
                    t_test_ratio, t_freeze, t_seed, t_small_model,
                ],
            )

            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            gr.Markdown("### Live Training Log")
            train_log = gr.Textbox(
                label="Log",
                lines=20,
                interactive=False,
                elem_classes="log-box",
            )
            gr.Markdown("### Training Results")
            with gr.Row():
                train_loss_plot = gr.Image(
                    label="Training vs Validation Loss",
                    type="pil",
                )
                train_predictions_plot = gr.Image(
                    label="Predictions vs Actuals",
                    type="pil",
                )
            train_result_summary = gr.Textbox(
                label="Result Summary",
                lines=8,
                interactive=False,
                elem_classes="status-box",
            )

            train_btn.click(
                fn=start_training,
                inputs=[
                    training_config_mode, site_dd,
                    t_num_images, t_epochs, t_batch, t_img_size,
                    t_lr, t_val_ratio, t_test_ratio,
                    t_freeze, t_seed,
                    t_small_model, t_save_drive,
                    roi_mode, selected_roi_state, roi_sample_size_state,
                ],
                outputs=[
                    train_log, train_loss_plot,
                    train_predictions_plot, train_result_summary,
                ],
            )

        # ===================================================================
        # TAB 4 – Results
        # ===================================================================
        with gr.Tab("4 – Results"):
            gr.Markdown("### Step 4 – Training Outputs",
                        elem_classes="section-header")
            refresh_btn = gr.Button("Load Results")
            with gr.Row():
                loss_plot = gr.Image(label="Training Loss Plot", type="pil")
                predictions_plot = gr.Image(label="Predictions vs Actuals", type="pil")
                summary   = gr.Textbox(
                    label="Summary",
                    lines=12,
                    interactive=False,
                    elem_classes="status-box",
                )
            refresh_btn.click(
                fn=get_training_outputs,
                outputs=[loss_plot, predictions_plot, summary],
            )

        # ===================================================================
        # TAB 5 – Inference
        # ===================================================================
        with gr.Tab("5 – Inference"):
            gr.Markdown("### Download USGS Images and Run Inference",
                        elem_classes="section-header")

            with gr.Row():
                with gr.Column():
                    inference_site_dd = gr.Dropdown(
                        choices=SITE_CHOICES,
                        label="USGS Site",
                        value=SITE_CHOICES[0],
                    )
                    with gr.Row():
                        inference_start_date = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value="2025-01-01",
                            placeholder="2025-01-01",
                        )
                        inference_end_date = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            value="2025-03-31",
                            placeholder="2025-03-31",
                        )
                    inference_max_images = gr.Slider(
                        minimum=10, maximum=500, value=100, step=10,
                        label="Max images to download for inference",
                    )
                    inference_param_checks = gr.CheckboxGroup(
                        choices=PARAM_CHOICES,
                        value=_default_param_selection(SITE_CHOICES[0]),
                        label="USGS Parameters to fetch and include in CSV",
                    )
                    inference_api_key = gr.Textbox(
                        label="NIMS API Key (optional – leave blank to try without)",
                        placeholder="Paste your api.waterdata.usgs.gov key here",
                        type="password",
                    )
                with gr.Column():
                    inference_model_path = gr.Textbox(
                        label="Model .pth path",
                        value=os.path.join(RESULTS_DIR, "best_model.pth"),
                        placeholder="/content/water_level_demo/results/best_model.pth",
                    )
                    inference_scaler_path = gr.Textbox(
                        label="Scaler .pkl path",
                        value=os.path.join(RESULTS_DIR, "scaler.pkl"),
                        placeholder="/content/water_level_demo/results/scaler.pkl",
                    )
                    inference_config_path = gr.Textbox(
                        label="Training config.json path",
                        value=os.path.join(RESULTS_DIR, "config.json"),
                        placeholder="/content/water_level_demo/results/config.json",
                    )
                    inference_labels_csv_path = gr.Textbox(
                        label="Labels CSV path",
                        value=DEFAULT_LABELS_CSV_PATH,
                        placeholder="/content/water_level_demo/data/labels.csv",
                    )
                    inference_img_size = gr.Number(
                        value=None,
                        precision=0,
                        label="Input image size override (blank = config.json)",
                    )
                    inference_batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=1,
                        step=1,
                        label="Batch size",
                    )

            inference_site_dd.change(
                fn=_default_param_selection,
                inputs=inference_site_dd,
                outputs=inference_param_checks,
            )

            inference_btn = gr.Button("Run Inference", variant="primary", size="lg")
            inference_status = gr.Textbox(
                label="Inference Status",
                lines=16,
                interactive=False,
                elem_classes="status-box",
            )
            inference_df = gr.Dataframe(
                label="Prediction Preview",
                interactive=False,
            )
            inference_csv = gr.File(label="Download inference_predictions.csv")
            with gr.Row():
                inference_timeseries_plot = gr.Image(
                    label="Predicted vs USGS Observed Time Series",
                    type="pil",
                )
                inference_scatter_plot = gr.Image(
                    label="Predicted vs True",
                    type="pil",
                )
                inference_error_plot = gr.Image(
                    label="Absolute Error Over Time",
                    type="pil",
                )

            inference_btn.click(
                fn=run_inference_handler,
                inputs=[
                    inference_model_path,
                    inference_scaler_path,
                    inference_config_path,
                    inference_labels_csv_path,
                    inference_site_dd,
                    inference_start_date,
                    inference_end_date,
                    inference_max_images,
                    inference_param_checks,
                    inference_api_key,
                    inference_img_size,
                    inference_batch_size,
                ],
                outputs=[
                    inference_status,
                    inference_df,
                    inference_csv,
                    inference_timeseries_plot,
                    inference_scatter_plot,
                    inference_error_plot,
                ],
            )

        # Footer
        gr.Markdown(
            """
            ---
            **Outputs** saved to `water_level_demo/results/`:
            `best_model.pth` · `scaler.pkl` · `training_history.csv` ·
            `training_loss_plot.png` · `config.json` · `test_results_<site>.csv` ·
            `predictions_vs_actuals_<site>.png` · `loss_curves_<site>.png` ·
            `inference/inference_predictions.csv`
            """
        )

    demo.launch(
        share=share,
        debug=debug,
        show_error=show_error,
    )
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    td.check_gpu()
    launch_gradio(share=False)
