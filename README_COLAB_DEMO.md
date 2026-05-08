# 💧 USGS Water Level Prediction – Colab Training Demo

> **EfficientNet Regression · Google Colab Free Tier · T4 GPU**

---

## What This Is

A self-contained Google Colab training demo adapted from the production `MAIN2.ipynb` pipeline.  
It strips AWS/S3/EC2 dependencies and wraps the same training logic in a **Gradio UI** optimized for Colab T4.

---

## Files in This Package

| File | Purpose |
|---|---|
| `Water_Level_Training_Demo.ipynb` | Main Colab notebook – run this |
| `train_demo.py` | Training module (adapted from MAIN2) |
| `app.py` | Gradio UI wrapping `train_demo.py` |
| `MAIN2.ipynb` | Original production pipeline (reference) |

---

## How to Run on Google Colab

### Step 1 – Upload to Colab

Option A (recommended): Upload all files to Google Drive and open the notebook from there.  
Option B: Upload directly to Colab's file panel (lost on disconnect).

### Step 2 – Enable GPU

In Colab: `Runtime → Change runtime type → T4 GPU → Save`

### Step 3 – Run cells in order

```
Cell 0  →  GPU check + folder setup
Cell 1  →  Install packages  (~2 min first time)
Cell 2  →  Copy train_demo.py and app.py to working dir
Cell 3  →  Launch Gradio UI  (opens a public share link)
```

### Step 4 – Use the Gradio UI

1. **Tab 1 – Dataset**: Upload your `labels.csv` and images ZIP (or enter a Drive path).  
   Click **Prepare Dataset** to validate.
2. **Tab 2 – ROI**: Select the site (Pinewood Road auto-fills the ROI). Click **Preview ROI Crop** to verify.
3. **Tab 3 – Train**: Adjust settings and click **Start Training**. Watch the live log.
4. **Tab 4 – Results**: Click **Load Results** to see the loss plot and summary.

---

## Dataset CSV Format

Your CSV must have:

| Column | Accepted names |
|---|---|
| Image path/filename | `image_path`, `dfile_path`, `filename`, `file`, `image` |
| Water level | `water_level`, `usgstrue_wl`, `gage_height`, `target`, `label` |
| Timestamp (optional) | `dt_pdatetime`, `timestamp`, `datetime`, `date_time` |

---

## Default Pinewood ROI

```
y1=951, x1=0, y2=1136, x2=1920
```
Crops the water surface region from a 1920×1080 camera image.

---

## Training Outputs

All saved to `/content/water_level_demo/results/`:

| File | Contents |
|---|---|
| `best_model.pth` | Model weights at best validation loss |
| `scaler.pkl` | StandardScaler fitted on training targets |
| `training_history.csv` | Per-epoch train/val loss |
| `training_loss_plot.png` | Loss curve chart |
| `config.json` | Full hyperparameter snapshot |
| `split_summary.csv` | Train/val/test counts |

Use **Cell 6** to download a ZIP of all results.

---

## Memory Tips (T4 · 15 GB VRAM)

| Backbone | Image size | Batch size | VRAM |
|---|---|---|---|
| EfficientNet-B3 | 384 | 4 | ~4–6 GB ✅ |
| EfficientNet-B3 | 512 | 4 | ~7–9 GB ⚠️ |
| EfficientNet-L2 | 512 | 8 | >15 GB ❌ OOM |

**If CUDA Out of Memory:** reduce batch size to 2 or image size to 224.

---

## What's NOT in v1 (coming in v2)

- Inference pipeline
- USGS API label matching
- R²/MAE/RMSE scatter plot
- Time-series prediction plot

---

## Architecture (matches production)

```
Input image
    ↓  ROI crop  (y1,x1,y2,x2)
    ↓  Resize    (configurable: 224/384/512/600 px)
    ↓  [Training only] ColorJitter + RandomPerspective
    ↓  ToTensor + ImageNet Normalize
    ↓  EfficientNet backbone (pretrained, partially frozen)
    ↓  Regression head: Linear(1024→512→128→1)
    ↓  Scalar output → inverse StandardScaler → water level (ft/m)
```
