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

# ---------------------------------------------------------------------------
# Paths – auto-detect Colab vs local
# ---------------------------------------------------------------------------
if os.path.exists("/content"):
    BASE_DIR = "/content/water_level_demo"
else:
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_level_demo")

DATA_DIR    = os.path.join(BASE_DIR, "data")
IMAGES_DIR  = os.path.join(DATA_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DRIVE_DIR   = "/content/drive/MyDrive/water_level_demo/results"

for d in [BASE_DIR, DATA_DIR, IMAGES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Global shared state
# ---------------------------------------------------------------------------
_state = {
    "mappings":   None,   # {abs_image_path: float water_level}
    "roi":        (951, 0, 1136, 1920),
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
        yield "Please select a site."
        return

    # Resolve selected parameter codes from checkbox labels
    code_map = {f"{code} – {label}": code
                for code, label in da.USGS_PARAMETERS.items()}
    param_codes = [code_map[s] for s in param_selection if s in code_map]

    if not param_codes:
        yield "Please select at least one USGS parameter."
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
    while True:
        try:
            line = _acq_queue.get(timeout=300)
        except queue.Empty:
            log_lines.append("Timeout – no output for 5 minutes.")
            break
        if line in ("__DONE_OK__", "__DONE_ERR__"):
            break
        log_lines.append(line)
        yield "\n".join(log_lines)

    t.join(timeout=10)
    yield "\n".join(log_lines)


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

def preview_roi_handler(y1, x1, y2, x2):
    roi = (int(y1), int(x1), int(y2), int(x2))
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
            f"Cropped: {cropped.width}x{cropped.height} px"
        )
        return orig, cropped, status
    except Exception as e:
        return None, None, f"Preview failed: {e}"


def site_roi_autofill(site_name):
    info = da.SITE_CATALOG.get(site_name)
    if info:
        y1, x1, y2, x2 = info["roi"]
        _state["roi"] = (y1, x1, y2, x2)
        # Build default param checkbox selections
        default_codes = info.get("default_params", ["00065"])
        default_sel = [f"{code} – {da.USGS_PARAMETERS.get(code, code)}"
                       for code in default_codes
                       if f"{code} – {da.USGS_PARAMETERS.get(code, code)}" in PARAM_CHOICES]
        return y1, x1, y2, x2, f"ROI auto-filled for {site_name}", default_sel
    return 0, 0, 1080, 1920, "No default ROI – set manually.", [PARAM_CHOICES[0]]


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
        _log_queue.put(f"Training error: {e}\n{traceback.format_exc()}")
        _log_queue.put("__DONE__")


def start_training(
    num_images, n_epochs, batch_size, img_size,
    learning_rate, val_ratio, test_ratio,
    freeze_ratio, seed,
    use_small_backbone, save_to_drive,
):
    mappings = _state.get("mappings")
    if not mappings:
        yield "No dataset loaded. Please run Data Acquisition first."
        return

    n = min(int(num_images), len(mappings))
    keys = list(mappings.keys())
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(keys), size=n, replace=False)
    sub = {keys[i]: mappings[keys[i]] for i in chosen}

    roi = _state.get("roi", (951, 0, 1136, 1920))
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
        yield "\n".join(log_lines)

    thread.join(timeout=5)
    yield "\n".join(log_lines)


def get_training_outputs():
    if not _train_result:
        return None, "No results yet. Run training first."

    plot_path = _train_result.get("plot_path")
    img = Image.open(plot_path) if plot_path and os.path.exists(plot_path) else None

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
        f"Scaler               : {_train_result.get('scaler_path', 'N/A')}\n"
        f"Loss plot            : {_train_result.get('plot_path', 'N/A')}\n"
        f"Config               : {_train_result.get('config_path', 'N/A')}\n"
    )
    return img, summary


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
.log-box    { font-family: monospace; font-size: 0.82em; background: #0f172a;
              color: #94a3b8; border-radius: 6px; padding: 8px; }
"""

PARAM_CHOICES = [f"{code} – {label}" for code, label in da.USGS_PARAMETERS.items()]
SITE_CHOICES  = list(da.SITE_CATALOG.keys())


def launch_gradio(share: bool = True, debug: bool = False):
    """Build and launch the Gradio app."""

    with gr.Blocks(title="Water Level Prediction – Training Demo") as demo:

        gr.Markdown(
            """
            # USGS Water Level Prediction – Training Demo
            ### HiVIS Image Download · USGS Sensor Data · EfficientNet Regression
            ---
            Follow the 4 tabs in order: **Acquire Data → Set ROI → Train → Results**
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
                        value=["00065 – Gage height (ft)"],
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
                outputs=acq_log,
            )
            acq_btn.click(
                fn=get_acquisition_summary,
                inputs=[],
                outputs=acq_summary,
            )

        # ===================================================================
        # TAB 2 – ROI Settings
        # ===================================================================
        with gr.Tab("2 – ROI"):
            gr.Markdown("### Step 2 – Region of Interest",
                        elem_classes="section-header")

            gr.Markdown(
                "The ROI is **auto-filled** from the site you selected in Tab 1. "
                "Adjust if needed, then click **Preview**."
            )

            with gr.Row():
                roi_y1 = gr.Number(value=951,  label="y1 (top)")
                roi_x1 = gr.Number(value=0,    label="x1 (left)")
                roi_y2 = gr.Number(value=1136, label="y2 (bottom)")
                roi_x2 = gr.Number(value=1920, label="x2 (right)")

            preview_btn = gr.Button("Preview ROI Crop")
            with gr.Row():
                orig_img    = gr.Image(label="Original Image",  type="pil")
                cropped_img = gr.Image(label="Cropped ROI",     type="pil")
            roi_status = gr.Textbox(label="ROI Status", interactive=False)

            def _save_roi(y1, x1, y2, x2):
                _state["roi"] = (int(y1), int(x1), int(y2), int(x2))

            for src in [roi_y1, roi_x1, roi_y2, roi_x2]:
                src.change(fn=_save_roi,
                           inputs=[roi_y1, roi_x1, roi_y2, roi_x2],
                           outputs=[])

            # Auto-fill ROI when site changes in Tab 1
            site_dd.change(
                fn=site_roi_autofill,
                inputs=site_dd,
                outputs=[roi_y1, roi_x1, roi_y2, roi_x2, roi_status, param_checks],
            )
            preview_btn.click(
                fn=preview_roi_handler,
                inputs=[roi_y1, roi_x1, roi_y2, roi_x2],
                outputs=[orig_img, cropped_img, roi_status],
            )

        # ===================================================================
        # TAB 3 – Training
        # ===================================================================
        with gr.Tab("3 – Train"):
            gr.Markdown("### Step 3 – Configure and Start Training",
                        elem_classes="section-header")

            with gr.Row():
                with gr.Column():
                    t_num_images = gr.Slider(50, 500, value=200, step=10,
                                            label="Number of images to train on")
                    t_epochs     = gr.Slider(1, 20, value=5, step=1,
                                            label="Epochs")
                    t_batch      = gr.Dropdown(choices=[2, 4, 8, 16], value=4,
                                              label="Batch size")
                    t_img_size   = gr.Dropdown(choices=[224, 384, 512, 600],
                                              value=384,
                                              label="Input image size (px)")
                with gr.Column():
                    t_lr         = gr.Number(value=2e-4, label="Learning rate",
                                            precision=6)
                    t_val_ratio  = gr.Number(value=0.15, label="Validation split",
                                            precision=2)
                    t_test_ratio = gr.Number(value=0.10, label="Test split",
                                            precision=2)
                    t_freeze     = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                            label="Backbone freeze ratio")
                    t_seed       = gr.Number(value=42, label="Random seed",
                                            precision=0)
                with gr.Column():
                    t_small_model = gr.Checkbox(
                        value=True,
                        label="Use EfficientNet-B3 (recommended – fits on T4 GPU)",
                    )
                    t_save_drive  = gr.Checkbox(
                        value=False,
                        label="Copy results to Google Drive",
                    )
                    gr.Markdown(
                        """
                        **GPU Memory tips (T4 – 15 GB):**
                        - B3 + 384px + batch 4 = ~4–6 GB OK
                        - L2 + 512px + batch 8 = likely OOM
                        - If out-of-memory: reduce batch or image size
                        """
                    )

            train_btn = gr.Button("Start Training", variant="primary", size="lg")
            gr.Markdown("### Live Training Log")
            train_log = gr.Textbox(
                label="Log",
                lines=20,
                interactive=False,
                elem_classes="log-box",
            )

            train_btn.click(
                fn=start_training,
                inputs=[
                    t_num_images, t_epochs, t_batch, t_img_size,
                    t_lr, t_val_ratio, t_test_ratio,
                    t_freeze, t_seed,
                    t_small_model, t_save_drive,
                ],
                outputs=train_log,
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
                summary   = gr.Textbox(
                    label="Summary",
                    lines=12,
                    interactive=False,
                    elem_classes="status-box",
                )
            refresh_btn.click(
                fn=get_training_outputs,
                outputs=[loss_plot, summary],
            )

        # Footer
        gr.Markdown(
            """
            ---
            **Outputs** saved to `water_level_demo/results/`:
            `best_model.pth` · `scaler.pkl` · `training_history.csv` ·
            `training_loss_plot.png` · `config.json`
            """
        )

    demo.launch(
        share=share,
        debug=debug,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
    )
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    td.check_gpu()
    launch_gradio(share=False)
