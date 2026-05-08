"""
app.py – Gradio Training UI for Water Level Prediction Demo
===========================================================
Designed to run inside a Google Colab notebook cell.
Launches a Gradio interface that wraps train_demo.py.

Sections:
  1. Dataset Upload (CSV + images ZIP or Drive path)
  2. Site / ROI Settings
  3. Training Hyperparameters
  4. Start Training + live log stream
  5. Results display (loss plot + summary)
"""

import os
import io
import zipfile
import shutil
import threading
import queue
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

# Import our training module
import train_demo as td

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = "/content/water_level_demo"
DATA_DIR     = os.path.join(BASE_DIR, "data")
IMAGES_DIR   = os.path.join(DATA_DIR, "images")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
DRIVE_DIR    = "/content/drive/MyDrive/water_level_demo/results"

for d in [BASE_DIR, DATA_DIR, IMAGES_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Global state (shared between Gradio event handlers)
# ---------------------------------------------------------------------------
_state = {
    "df":         None,   # loaded DataFrame
    "img_col":    None,
    "target_col": None,
    "time_col":   None,
    "mappings":   None,   # {abs_path: float}
    "roi":        None,   # (y1, x1, y2, x2)
}

# ---------------------------------------------------------------------------
# Helper: extract ZIP to images folder
# ---------------------------------------------------------------------------

def _extract_zip(zip_path: str, dest_dir: str) -> int:
    """Extract zip, return count of image files extracted."""
    os.makedirs(dest_dir, exist_ok=True)
    count = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            lower = member.lower()
            if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                # Flatten nested directories
                filename = os.path.basename(member)
                if not filename:
                    continue
                dest = os.path.join(dest_dir, filename)
                with zf.open(member) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                count += 1
    return count


# ---------------------------------------------------------------------------
# Section 1 – Dataset Upload
# ---------------------------------------------------------------------------

def prepare_dataset(csv_file, zip_file, drive_path, max_images, seed):
    """
    Gradio handler: load CSV + images, build image-label mapping.
    Returns status message.
    """
    # ── Load CSV ──────────────────────────────────────────────────────────
    if csv_file is None:
        return "❌ Please upload a CSV labels file."

    try:
        # Gradio 4.x: gr.File yields a NamedString (the filepath itself),
        # not a file object, so we can use it directly as a path.
        csv_path = csv_file if isinstance(csv_file, str) else csv_file.name
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"❌ Could not read CSV: {e}"

    try:
        img_col, target_col, time_col = td.detect_columns(df)
    except ValueError as e:
        return str(e)

    _state["df"]         = df
    _state["img_col"]    = img_col
    _state["target_col"] = target_col
    _state["time_col"]   = time_col

    # ── Resolve image directory ───────────────────────────────────────────
    if zip_file is not None:
        try:
            zip_path = zip_file if isinstance(zip_file, str) else zip_file.name
            n = _extract_zip(zip_path, IMAGES_DIR)
            img_dir = IMAGES_DIR
            src_msg = f"ZIP extracted ({n} images → {IMAGES_DIR})"
        except Exception as e:
            return f"❌ ZIP extraction failed: {e}"
    elif drive_path and drive_path.strip():
        img_dir = drive_path.strip()
        if not os.path.isdir(img_dir):
            return f"❌ Google Drive path not found: {img_dir}"
        src_msg = f"Using Drive folder: {img_dir}"
    else:
        return "❌ Please upload an images ZIP OR enter a Google Drive folder path."

    # ── Build mapping ─────────────────────────────────────────────────────
    try:
        mappings = td.build_image_label_mapping(
            df, img_col, target_col,
            image_dir=img_dir,
            roi=None,                  # ROI applied at Dataset level
            max_images=int(max_images),
            seed=int(seed),
        )
    except ValueError as e:
        return str(e)

    _state["mappings"] = mappings

    targets = list(mappings.values())
    return (
        f"✅ Dataset ready!\n"
        f"   Source: {src_msg}\n"
        f"   Columns detected: image={img_col}, target={target_col}"
        + (f", time={time_col}" if time_col else "") + "\n"
        f"   Images matched: {len(mappings)}\n"
        f"   Water level range: {min(targets):.3f} – {max(targets):.3f}\n"
        f"   Water level mean:  {float(np.mean(targets)):.3f}"
    )


# ---------------------------------------------------------------------------
# Section 2 – ROI Settings
# ---------------------------------------------------------------------------

def update_roi(y1, x1, y2, x2):
    """Store ROI in global state."""
    try:
        _state["roi"] = (int(y1), int(x1), int(y2), int(x2))
        return f"✅ ROI set: y1={y1}, x1={x1}, y2={y2}, x2={x2}"
    except Exception as e:
        return f"❌ Invalid ROI values: {e}"


def preview_roi_handler(y1, x1, y2, x2):
    """
    Pick first available image, show original + cropped ROI side-by-side.
    Returns (original_pil, cropped_pil, status_str)
    """
    roi = (int(y1), int(x1), int(y2), int(x2))
    _state["roi"] = roi

    # Find a sample image
    mappings = _state.get("mappings")
    if not mappings:
        # Fall back to any image in IMAGES_DIR
        all_imgs = [
            os.path.join(IMAGES_DIR, f)
            for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not all_imgs:
            return None, None, "❌ No images found. Run 'Prepare Dataset' first."
        img_path = all_imgs[0]
    else:
        img_path = list(mappings.keys())[0]

    try:
        orig, cropped = td.preview_roi(img_path, roi)
        status = (
            f"✅ ROI preview: original {orig.width}×{orig.height} px\n"
            f"   Crop region: y={y1}–{y2}, x={x1}–{x2}  "
            f"→ {cropped.width}×{cropped.height} px"
        )
        return orig, cropped, status
    except Exception as e:
        return None, None, f"❌ Preview failed: {e}"


def site_changed(site_name):
    """Auto-fill ROI defaults when site dropdown changes."""
    defaults = td.DEFAULT_ROI.get(site_name)
    if defaults:
        y1, x1, y2, x2 = defaults
        return y1, x1, y2, x2, f"ROI auto-filled for {site_name}"
    return 0, 0, 1080, 1920, "No default ROI for this site – please set manually."


# ---------------------------------------------------------------------------
# Section 3 + 4 – Training
# ---------------------------------------------------------------------------

# Running training in a thread so Gradio can stream logs
_log_queue: queue.Queue = queue.Queue()
_train_result = {}


def _training_thread(kwargs):
    """Target for the training thread. Pushes log lines to _log_queue."""
    def _log_cb(msg):
        _log_queue.put(msg)

    try:
        result = td.train_model(**kwargs, log_callback=_log_cb)
        _train_result.update(result)
        _log_queue.put("__DONE__")
    except Exception as e:
        _log_queue.put(f"❌ Training error: {e}\n{traceback.format_exc()}")
        _log_queue.put("__DONE__")


def start_training(
    num_images, n_epochs, batch_size, img_size,
    learning_rate, val_ratio, test_ratio,
    freeze_ratio, seed,
    use_small_backbone, save_to_drive,
):
    """Gradio click handler for 'Start Training' button. Streams log text."""
    mappings = _state.get("mappings")
    if not mappings:
        yield "❌ No dataset prepared. Please run 'Prepare Dataset' first."
        return

    # Sub-sample to requested count
    n = min(int(num_images), len(mappings))
    keys = list(mappings.keys())
    rng  = np.random.default_rng(int(seed))
    chosen_keys = rng.choice(len(keys), size=n, replace=False)
    sub_mappings = {keys[i]: mappings[keys[i]] for i in chosen_keys}

    # Read ROI from the global _state dict (set by the ROI tab)
    roi = _state.get("roi", (951, 0, 1136, 1920))

    # Choose backbone
    if use_small_backbone:
        backbone = "tf_efficientnet_b3.ns_jft_in1k"
    else:
        backbone = "tf_efficientnet_l2.ns_jft_in1k"

    kwargs = dict(
        mappings         = sub_mappings,
        roi              = roi,
        results_dir      = RESULTS_DIR,
        num_epochs       = int(n_epochs),
        batch_size       = int(batch_size),
        input_img_size   = int(img_size),
        learning_rate    = float(learning_rate),
        val_ratio        = float(val_ratio),
        test_ratio       = float(test_ratio),
        param_freeze_ratio = float(freeze_ratio),
        seed             = int(seed),
        backbone_name    = backbone,
        save_to_drive    = bool(save_to_drive),
        drive_dir        = DRIVE_DIR,
    )

    # Clear old queue
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
            log_lines.append("⚠️  Training timeout – no output for 2 minutes.")
            break
        if line == "__DONE__":
            break
        log_lines.append(line)
        yield "\n".join(log_lines)

    thread.join(timeout=5)
    yield "\n".join(log_lines)


def get_training_outputs():
    """Return (loss_plot_image, summary_text) after training finishes."""
    if not _train_result:
        return None, "No training results yet. Run training first."

    plot_path = _train_result.get("plot_path")
    img = Image.open(plot_path) if plot_path and os.path.exists(plot_path) else None

    t = _train_result.get("total_time_s", 0)
    h, rem = divmod(int(t), 3600)
    m, s   = divmod(rem, 60)

    # Guard: best_val_loss may be float or missing
    bvl = _train_result.get("best_val_loss", None)
    bvl_str = f"{bvl:.4f}" if isinstance(bvl, (int, float)) else "N/A"

    summary = (
        f"📊 Training Summary\n"
        f"{'─'*40}\n"
        f"Best validation loss : {bvl_str}\n"
        f"Total training time  : {h}h {m}m {s}s\n"
        f"Best model saved to  : {_train_result.get('best_model_path', 'N/A')}\n"
        f"Scaler saved to      : {_train_result.get('scaler_path', 'N/A')}\n"
        f"Loss plot saved to   : {_train_result.get('plot_path', 'N/A')}\n"
        f"Config saved to      : {_train_result.get('config_path', 'N/A')}\n"
    )
    return img, summary


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Inter', sans-serif; }
.gr-button-primary { background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; }
.section-header { 
    font-size: 1.1em; font-weight: 700; color: #1e40af;
    border-bottom: 2px solid #bfdbfe; padding-bottom: 4px; margin-bottom: 8px;
}
.status-box { font-family: monospace; font-size: 0.85em; }
"""

def launch_gradio(share: bool = True, debug: bool = False):
    """Build and launch the Gradio app."""

    with gr.Blocks(
        css=CSS,
        title="💧 Water Level Training Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="sky",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 💧 USGS Water Level Prediction – Training Demo
            ### EfficientNet Regression · Google Colab T4 Edition
            ---
            **Instructions:**  
            1. Upload your CSV labels file and image ZIP (or enter a Drive path).  
            2. Click **Prepare Dataset** to validate and index your data.  
            3. Adjust the ROI and preview it on a sample image.  
            4. Tune training settings and click **Start Training**.  
            5. Monitor the live log and download outputs when done.
            """
        )

    # ROI is stored in the _state Python dict by the ROI tab handlers.
    # No gr.State or cross-tab Gradio component references needed.
    # Default = Pinewood Road ROI.
    _state.setdefault("roi", (951, 0, 1136, 1920))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1 – Dataset Upload
    # ═══════════════════════════════════════════════════════════════════
    with gr.Tab("📂 1 · Dataset"):
        gr.Markdown("### Dataset Upload", elem_classes="section-header")
        with gr.Row():
            with gr.Column(scale=1):
                csv_upload = gr.File(
                    label="Upload CSV labels file",
                    file_types=[".csv"],
                )
                zip_upload = gr.File(
                    label="Upload images ZIP file",
                    file_types=[".zip"],
                )
                drive_path_in = gr.Textbox(
                    label="— OR — Google Drive image folder path",
                    placeholder="/content/drive/MyDrive/your_images/",
                )
            with gr.Column(scale=1):
                max_images_slider = gr.Slider(
                    minimum=50, maximum=500, value=200, step=10,
                    label="Max images to use",
                )
                seed_in = gr.Number(value=42, label="Random seed", precision=0)
                prepare_btn = gr.Button("🚀 Prepare Dataset", variant="primary")
                dataset_status = gr.Textbox(
                    label="Status",
                    lines=8,
                    interactive=False,
                    elem_classes="status-box",
                )

        prepare_btn.click(
            fn=prepare_dataset,
            inputs=[csv_upload, zip_upload, drive_path_in,
                    max_images_slider, seed_in],
            outputs=dataset_status,
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2 – Site / ROI Settings
    # ═══════════════════════════════════════════════════════════════════
    with gr.Tab("🔲 2 · ROI"):
        gr.Markdown("### Site & ROI Settings", elem_classes="section-header")
        with gr.Row():
            site_dd = gr.Dropdown(
                choices=["Pinewood Road", "Custom"],
                value="Pinewood Road",
                label="Site",
            )
            roi_status = gr.Textbox(label="ROI Status", interactive=False)

        with gr.Row():
            roi_y1 = gr.Number(value=951,  label="y1 (top crop)")
            roi_x1 = gr.Number(value=0,    label="x1 (left crop)")
            roi_y2 = gr.Number(value=1136, label="y2 (bottom crop)")
            roi_x2 = gr.Number(value=1920, label="x2 (right crop)")

        preview_btn = gr.Button("🖼️ Preview ROI Crop")
        with gr.Row():
            orig_img    = gr.Image(label="Original Image", type="pil")
            cropped_img = gr.Image(label="Cropped ROI",    type="pil")
        preview_status = gr.Textbox(label="Preview Status", interactive=False)

        # Write ROI values to _state dict whenever a number changes
        def _save_roi_to_state(y1, x1, y2, x2):
            _state["roi"] = (int(y1), int(x1), int(y2), int(x2))

        for _src in [roi_y1, roi_x1, roi_y2, roi_x2]:
            _src.change(
                fn=_save_roi_to_state,
                inputs=[roi_y1, roi_x1, roi_y2, roi_x2],
                outputs=[],
            )

        site_dd.change(
            fn=site_changed,          # existing function – returns y1,x1,y2,x2,status
            inputs=site_dd,
            outputs=[roi_y1, roi_x1, roi_y2, roi_x2, roi_status],
        )
        preview_btn.click(
            fn=preview_roi_handler,
            inputs=[roi_y1, roi_x1, roi_y2, roi_x2],
            outputs=[orig_img, cropped_img, preview_status],
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3+4 – Training
    # ═══════════════════════════════════════════════════════════════════
    with gr.Tab("🏋️ 3 · Train"):
        gr.Markdown("### Training Settings", elem_classes="section-header")
        with gr.Row():
            with gr.Column():
                t_num_images = gr.Slider(50, 500, value=200, step=10,
                                        label="Number of images")
                t_epochs     = gr.Slider(1, 20, value=5, step=1,
                                        label="Epochs")
                t_batch      = gr.Dropdown(choices=[2, 4, 8, 16], value=4,
                                          label="Batch size")
                t_img_size   = gr.Dropdown(
                    choices=[224, 384, 512, 600], value=384,
                    label="Input image size (px)",
                )
            with gr.Column():
                t_lr         = gr.Number(value=2e-4, label="Learning rate",
                                        precision=6)
                t_val_ratio  = gr.Number(value=0.15, label="Validation split ratio",
                                        precision=2)
                t_test_ratio = gr.Number(value=0.10, label="Test split ratio",
                                        precision=2)
                t_freeze     = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                        label="Backbone freeze ratio")
                t_seed       = gr.Number(value=42, label="Random seed", precision=0)
            with gr.Column():
                t_small_model  = gr.Checkbox(
                    value=True,
                    label="✅ Use EfficientNet-B3 (lighter, recommended for T4)",
                )
                t_save_drive   = gr.Checkbox(
                    value=False,
                    label="☁️ Copy outputs to Google Drive",
                )
                gr.Markdown(
                    """
                    > **Memory tips (T4 · 15 GB VRAM):**  
                    > - B3 + size 384 + batch 4 → ~4–6 GB ✅  
                    > - L2 + size 512 + batch 8 → may OOM ⚠️  
                    > - If CUDA OOM: reduce batch size or image size.
                    """
                )

        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

        gr.Markdown("### Live Training Log")
        train_log = gr.Textbox(
            label="Log output",
            lines=20,
            interactive=False,
            elem_classes="status-box",
        )

        # streaming=True is required for generator-based (yield) handlers
        train_btn.click(
            fn=start_training,
            inputs=[
                t_num_images, t_epochs, t_batch, t_img_size,
                t_lr, t_val_ratio, t_test_ratio,
                t_freeze, t_seed,
                t_small_model, t_save_drive,
                # ROI is read from _state dict inside start_training
            ],
            outputs=train_log,
            streaming=True,
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5 – Outputs
    # ═══════════════════════════════════════════════════════════════════
    with gr.Tab("📊 4 · Results"):
        gr.Markdown("### Training Outputs", elem_classes="section-header")
        refresh_btn = gr.Button("🔄 Load Results")
        with gr.Row():
            loss_plot_out = gr.Image(label="Training Loss Plot", type="pil")
            summary_out   = gr.Textbox(
                label="Summary",
                lines=12,
                interactive=False,
                elem_classes="status-box",
            )
        refresh_btn.click(
            fn=get_training_outputs,
            outputs=[loss_plot_out, summary_out],
        )

    # ── Footer ─────────────────────────────────────────────────────────
    gr.Markdown(
        """
        ---
        **Output files** are saved to `/content/water_level_demo/results/`:
        `best_model.pth` · `scaler.pkl` · `training_history.csv` · 
        `training_loss_plot.png` · `config.json` · `split_summary.csv`

        *Inference pipeline and time-series plots will be added in v2.*
        """
    )

    demo.launch(share=share, debug=debug)
    return demo


# ---------------------------------------------------------------------------
# Entry point (called from notebook)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick GPU check on startup
    td.check_gpu()
    launch_gradio(share=True)
