# USGS Water Level Prediction – Training Demo

> **EfficientNet-based regression pipeline** that predicts water levels from USGS HiVIS camera images.  
> Designed to run as a **local Gradio app** or on **Google Colab (T4 GPU)**.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Full Workflow](#full-workflow)
   - [Step 1 – Data Acquisition](#step-1--data-acquisition)
   - [Step 2 – ROI Setting](#step-2--roi-setting)
   - [Step 3 – Training](#step-3--training)
   - [Step 4 – Results](#step-4--results)
5. [APIs Used](#apis-used)
6. [Supported Sites](#supported-sites)
7. [USGS Parameter Codes](#usgs-parameter-codes)
8. [Model Architecture](#model-architecture)
9. [Training Pipeline Deep-Dive](#training-pipeline-deep-dive)
10. [Output Files](#output-files)
11. [Running Locally](#running-locally)
12. [Running on Google Colab](#running-on-google-colab)
13. [Dependencies](#dependencies)
14. [Known Limitations](#known-limitations)

---

## Project Overview

This project trains a **deep learning regression model** to predict water surface levels from USGS streamgage camera images. Instead of manually uploading images and CSV files, the app automates the entire data pipeline:

```
Select Site + Date Range
        ↓
Download images from USGS HiVIS (NIMS API)
        ↓
Fetch water level & precipitation from USGS NWIS (waterservices API)
        ↓
Auto-join images + sensor readings into labels.csv (±15 min tolerance)
        ↓
Train EfficientNet regression model with live loss log
        ↓
Save model checkpoint, scaler, and loss plot
```

The model learns to **look at a camera image and estimate the current water level in feet**, which enables real-time inference without needing live sensor data at inference time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio Web UI (app.py)                   │
│                                                                 │
│  Tab 1: Acquire Data  │  Tab 2: ROI  │  Tab 3: Train  │ Tab 4  │
└────────┬──────────────┴──────┬────────┴──────┬─────────┴────────┘
         │                     │               │
         ▼                     │               ▼
┌─────────────────────┐        │    ┌─────────────────────┐
│  data_acquisition.py│        │    │    train_demo.py     │
│                     │        │    │                     │
│  - NIMS API calls   │        │    │  - Dataset class    │
│  - NWIS API calls   │        │    │  - EfficientNet     │
│  - Image download   │        │    │  - Training loop    │
│  - CSV building     │        │    │  - StandardScaler   │
└──────────┬──────────┘        │    └─────────────────────┘
           │                   │
           ▼                   ▼
    water_level_demo/
    ├── data/
    │   ├── images/       ← downloaded HiVIS JPGs
    │   └── labels.csv    ← image_path + water_level + precipitation
    └── results/
        ├── best_model.pth
        ├── scaler.pkl
        └── training_loss_plot.png
```

---

## Project Structure

```
Water-prediction-colab-version/
│
├── app.py                        # Gradio UI – main entry point
├── data_acquisition.py           # HiVIS + NWIS data pipeline
├── train_demo.py                 # ML training module
├── Water_Level_Training_Demo.ipynb  # Colab notebook wrapper
├── MAIN2.ipynb                   # Production training pipeline (S3-based)
├── .gitignore                    # Excludes images, models, cache
└── README.md                     # This file
```

---

## Full Workflow

### Step 1 – Data Acquisition

**File:** `data_acquisition.py` → called by `app.py` Tab 1

The user selects:
- **USGS Site** – dropdown of 10 pre-configured sites
- **Start Date / End Date** – e.g. `2025-01-01` to `2025-03-31`
- **Max images** – slider 50–500
- **USGS Parameters** – checkboxes (gage height, precipitation, etc.)
- **NIMS API Key** *(optional)*

The pipeline runs 5 steps automatically:

#### [1/5] Camera Discovery (NIMS API)
```
GET https://api.waterdata.usgs.gov/nims/v0/cameras
```
- Downloads the full list of ~1,171 USGS HiVIS cameras
- Filters locally by `camId` (exact match) or `nwisId` fallback
- Returns camera metadata including `overlayDir` (S3 image base URL)

> ⚠️ **Note:** The NIMS `site_no` query parameter does not actually filter — it always returns all cameras. The code filters locally.

#### [2/5] List Available Images (NIMS listFiles)
```
GET https://api.waterdata.usgs.gov/nims/v0/listFiles
    ?camId={camId}&after={start_date}&before={end_date}&limit={max}
```
- Returns a list of filename strings, e.g.:
  ```
  VA_Little_Neck_Creek_at_Pinewood_Road_at_Virginia_Beach___2025-01-02T23-54-21Z.jpg
  ```
- Timestamps are parsed from the filename using the format: `{camId}___YYYY-MM-DDTHH-MM-SSZ.jpg`

#### [3/5] Download Images
- Downloads each image from the S3 `overlayDir` URL
- Skips already-cached files (incremental)
- Stores to `water_level_demo/data/images/`

#### [4/5] Fetch USGS Sensor Data (NWIS Instantaneous Values)
```
GET https://waterservices.usgs.gov/nwis/iv/
    ?format=json&sites={site_no}&parameterCd={codes}&startDT=...&endDT=...
```
- Returns time-series at ~15-minute intervals
- Multiple parameters are merged on nearest timestamp
- Returns a DataFrame with columns like `62620_Estuary/ocean_water_surface_elevation_NAVD88`

#### [5/5] Build Labels CSV
- Joins image timestamps with USGS sensor readings using `pd.merge_asof` with a **15-minute tolerance**
- Renames the primary water level column to `water_level` for training compatibility
- Priority: `62620` (tidal) → `00065` (gage height) → `00060` (discharge)
- Saves to `water_level_demo/data/labels.csv`

**CSV Schema:**

| Column | Description |
|---|---|
| `image_path` | Absolute path to the downloaded image |
| `timestamp` | UTC datetime the image was captured |
| `water_level` | Primary target — water surface elevation (ft) |
| `00045_Precipitation` | Precipitation (in), if selected |
| `datetime` | USGS sensor reading timestamp |

---

### Step 2 – ROI Setting

**File:** `app.py` Tab 2, calls `train_demo.preview_roi()`

The **Region of Interest (ROI)** crops the camera image to focus on the water surface, removing sky, vegetation, and camera housing. Defined as `(y1, x1, y2, x2)` pixel coordinates.

- ROI **auto-fills** from the site catalog when you select a site
- A sample image is displayed before and after cropping for visual verification
- ROI is stored in `_state["roi"]` and passed to the training Dataset

**Default ROI for Pinewood Road site:** `(951, 0, 1136, 1920)`  
→ Crops to a 185px tall horizontal strip across the full 1920px width

---

### Step 3 – Training

**File:** `train_demo.py` → `train_model()`

Configurable parameters in the UI:

| Parameter | Default | Description |
|---|---|---|
| Number of images | 200 | How many of the downloaded images to use |
| Epochs | 5 | Training iterations over the dataset |
| Batch size | 4 | Images per gradient step |
| Image size | 384px | Input resolution to the model |
| Learning rate | 2e-4 | Initial learning rate (same as MAIN2.ipynb) |
| Validation split | 15% | Fraction held out for validation |
| Test split | 10% | Fraction held out for final evaluation |
| Backbone freeze ratio | 0.7 | Fraction of backbone layers frozen (0=all trainable) |
| Random seed | 42 | For reproducibility |
| EfficientNet-B3 mode | ✅ | Lighter model recommended for T4 GPU |

#### Training Loop Detail

```
1. Load CSV → detect image_path and water_level columns
2. Build {image_path: water_level} mapping
3. StandardScaler fit on training targets → saved as scaler.pkl
4. Split: train / val / test (stratified by water level)
5. For each epoch:
   a. Forward pass → EfficientNet backbone → regression head → predicted water level
   b. MSE Loss
   c. Backward pass + Adam optimizer step
   d. Validation loss logged to queue → streamed to Gradio log box
   e. If val_loss improved → save best_model.pth
6. Early stopping if no improvement for 3 epochs
7. Plot train vs val loss → training_loss_plot.png
```

#### Data Augmentation (training only)

```python
ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
RandomPerspective(distortion_scale=0.2, p=0.3)
Resize(input_img_size)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet
```

No augmentation during validation/test (only Resize + Normalize).

---

### Step 4 – Results

Displays:
- **Loss plot** – train vs validation MSE per epoch
- **Summary text** – best epoch, best val loss, output file paths

---

## APIs Used

### 1. USGS NIMS API (HiVIS Cameras)
**Base URL:** `https://api.waterdata.usgs.gov/nims/v0`

| Endpoint | Purpose |
|---|---|
| `GET /cameras` | List all ~1,171 HiVIS cameras |
| `GET /listFiles?camId=...&after=...&before=...&limit=...` | List image filenames for a camera in a date range |

Images are stored on S3:
```
https://usgs-nims-images.s3.amazonaws.com/overlay/{camId}/{filename}
```

**API Key:** Optional — sign up at https://apiwaterdata.usgs.gov/signup

### 2. USGS NWIS Instantaneous Values API
**Base URL:** `https://waterservices.usgs.gov/nwis/iv/`

```
GET /nwis/iv/?format=json&sites={site_no}&parameterCd={codes}&startDT=...&endDT=...
```

Returns JSON with nested time-series at ~15-min intervals.

---

## Supported Sites

| Site Name | NIMS nwisId | Default Parameters |
|---|---|---|
| VA Little Neck Creek (Pinewood Rd, VA Beach) | `0204295505` | `62620`, `00045` |
| VA Mechumps Creek (Hill Carter Pkwy, Ashland) | `0167300055` | `00065`, `00045` |
| VA Bailey Creek (Dock Landing Rd, Chesapeake) | `0204288905` | `62620`, `00045` |
| VA James River at Buchanan | `02019500` | `00065`, `00060` |
| VA Blackwater River at Franklin | `02050000` | `00065`, `00045` |
| VA Conveyance Channel (Ramsgate Ln, Great Bridge) | `0204309906` | `62620`, `00045` |
| NY Neversink River at Godeffroy | `01435000` | `00065`, `00045` |
| CA Sacramento River at Freeport | `11447650` | `00065`, `00045` |
| SC Congaree River at Columbia | `02169500` | `00065`, `00060` |
| NC McMullen Creek at Charlotte | `02146300` | `00065`, `00045` |

> **Tidal sites** (Pinewood, Bailey Creek, Conveyance Channel) use parameter `62620` (estuary/ocean water surface elevation, NAVD88) instead of `00065` (gage height).

---

## USGS Parameter Codes

| Code | Description | Unit | Sites |
|---|---|---|---|
| `00065` | Gage height | ft | Stream/river gauges |
| `62620` | Estuary/ocean water surface elevation (NAVD88) | ft | Tidal/coastal sites |
| `00045` | Precipitation | in | Most sites |
| `00060` | Discharge (streamflow) | cfs | River gauges |
| `00010` | Water temperature | °C | Select sites |
| `00300` | Dissolved oxygen | mg/L | Select sites |

---

## Model Architecture

```
Input Image (H×W×3)
       ↓
  ROI Crop (y1:y2, x1:x2)
       ↓
  Resize to (img_size × img_size)
       ↓
  ImageNet Normalize
       ↓
┌──────────────────────────────────┐
│   EfficientNet Backbone (timm)   │
│   tf_efficientnet_b3.ns_jft_in1k │
│   (or l2 for production)         │
│   ~48 MB weights                 │
│   Pre-trained on JFT-300M        │
└──────────────┬───────────────────┘
               │  Feature vector (1536-d for B3)
               ▼
        Linear(1536 → 1)
               ↓
     Scalar water level prediction (ft)
```

**Loss:** Mean Squared Error (MSE)  
**Optimizer:** Adam  
**Scheduler:** ReduceLROnPlateau (patience=2, factor=0.5)  
**Targets:** Scaled with `StandardScaler` before training, inverse-transformed at inference

---

## Training Pipeline Deep-Dive

### `train_demo.py` Key Functions

| Function | Description |
|---|---|
| `detect_columns(df)` | Auto-detects `image_path` and `water_level` columns from CSV |
| `build_image_label_mapping()` | Creates `{abs_path: float}` dict for the Dataset |
| `WaterLevelDataset` | PyTorch Dataset with ROI crop + augmentation |
| `EfficientNetRegressor` | Model class wrapping `timm` backbone + regression head |
| `train_model()` | Full training loop with checkpointing, logging, early stopping |
| `preview_roi()` | Returns PIL images (original + cropped) for Tab 2 preview |
| `check_gpu()` | Prints GPU/CPU status on startup |

### Mixed Precision Training
- Uses `torch.amp.GradScaler` and `torch.amp.autocast` when CUDA is available
- Automatically disabled on CPU (MPS on Apple Silicon not fully supported)

### Target Scaling
```python
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
# Model predicts scaled values
# At inference: scaler.inverse_transform(prediction)
```
Scaler is saved as `scaler.pkl` and must be bundled with `best_model.pth` for deployment.

---

## Output Files

All saved to `water_level_demo/results/`:

| File | Description |
|---|---|
| `best_model.pth` | PyTorch model weights (best validation loss epoch) |
| `scaler.pkl` | Fitted `StandardScaler` for target inverse-transform |
| `training_loss_plot.png` | Train vs val MSE per epoch |
| `training_history.csv` | Per-epoch loss values |
| `config.json` | Training hyperparameters snapshot |
| `split_summary.csv` | Train/val/test split statistics |

---

## Running Locally

### Prerequisites
- Python 3.10+
- macOS / Linux (Windows untested)
- No GPU required (CPU training is slow but functional)

### Setup

```bash
# Clone the repo
git clone https://github.com/yashsanap14/Water-prediction-colab-version.git
cd Water-prediction-colab-version

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision timm gradio scikit-learn pandas numpy matplotlib requests tqdm Pillow
```

### Launch

```bash
python app.py
```

Opens at: **http://127.0.0.1:7860**

---

## Running on Google Colab

```python
# Cell 1: Clone and install
!git clone https://github.com/yashsanap14/Water-prediction-colab-version.git
%cd Water-prediction-colab-version
!pip install timm gradio requests -q

# Cell 2: Launch with public share link
import sys
sys.path.insert(0, '.')
import app
app.launch_gradio(share=True)
```

> Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU) for fast training.
> 
> **Recommended settings on T4:**
> - EfficientNet-B3 ✅
> - Image size: 384px
> - Batch size: 4
> - Epochs: 5–10

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | Deep learning framework |
| `timm` | Pre-trained EfficientNet models |
| `gradio` | Web UI (requires v4+) |
| `scikit-learn` | StandardScaler, train/test split |
| `pandas` | CSV handling, time-series merging |
| `numpy` | Numerical operations |
| `matplotlib` | Loss plot generation |
| `requests` | NIMS + NWIS API calls |
| `tqdm` | Progress bars in training |
| `Pillow` | Image preview in Gradio |

---

## Known Limitations

| Limitation | Details |
|---|---|
| **CPU training speed** | ~10–50× slower than T4 GPU. For large datasets, use Colab. |
| **NIMS date filtering** | The `site_no` query param on the NIMS cameras endpoint does not filter — all 1,171 cameras are returned and filtered locally. |
| **15-min matching window** | Images captured more than 15 minutes from a USGS sensor reading will not be matched and are excluded from training. |
| **Tidal sites** | Coastal sites (Pinewood, Bailey Creek) measure `62620` (tidal elevation), not `00065` (gage height). Selecting the wrong parameter returns no data. |
| **No inference UI** | This app is training-only. Inference (predicting from a new image) requires a separate script using `best_model.pth` + `scaler.pkl`. |
| **NIMS API changes** | The USGS NIMS API is versioned as `v0` and may change. The verified working endpoint as of May 2026 is documented in `data_acquisition.py`. |

---

## Technical Reference

### Timestamp Matching Logic

```python
# HiVIS filename → datetime
# "VA_Little_Neck_Creek___2025-01-02T23-54-21Z.jpg"
#                         ^^^^^^^^^^^^ ^^^^^^^^
#                         date part    time part (hyphens as separators)

ts_part = "2025-01-02T23-54-21Z"
# Normalize: replace hyphens in time portion with colons
→ "2025-01-02T23:54:21+00:00"  # UTC
```

```python
# Merge images with USGS readings
pd.merge_asof(
    img_df.sort_values("dt_image"),
    sensor_df.sort_values("datetime"),
    left_on="dt_image",
    right_on="datetime",
    tolerance=pd.Timedelta("15min"),
    direction="nearest",
)
```

### NIMS Image URL Construction
```
overlayDir + filename
= "https://usgs-nims-images.s3.amazonaws.com/overlay/{camId}/"
  + "{camId}___YYYY-MM-DDTHH-MM-SSZ.jpg"
```

---

*Built for the USGS Water Data for the Nation (WDFN) modernization initiative.*  
*GitHub: [yashsanap14/Water-prediction-colab-version](https://github.com/yashsanap14/Water-prediction-colab-version)*
