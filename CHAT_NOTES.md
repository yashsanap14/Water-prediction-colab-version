# Project Chat Notes

This file summarizes the repository work and run instructions discussed in the chat.

## Repository Updates Completed

The following issues and features were addressed:

- Made `Water_Level_Training_Demo.ipynb` repo-relative and removed forced Google Drive mounting.
- Fixed the acquisition UI so the app does not call `get_acquisition_summary()` immediately after starting the background acquisition thread.
- Removed the hard-coded default water parameter `00065`; defaults now come from the selected site catalog.
- Updated `data_acquisition.py` so generated training CSVs require a valid numeric `water_level` target.
- Added safer training validation in `train_demo.py` before `train_test_split`.
- Updated README ROI coordinate documentation to use `(x1, y1, x2, y2)`.
- Added `requirements.txt`.
- Added ROI training image mode:
  - `Whole image`
  - `ROI cropped image`
- Added random sample image loading and ROI coordinate selection.
- Added Train tab configuration mode:
  - `Best configuration for training`
  - `Manual setup`
- Updated Best Configuration for the Colab-scaled app:
  - EfficientNet-B3
  - image size `384`
  - batch size `8`
  - fallback batch size `4`
  - learning rate `1e-4`
  - epochs `12`
  - split `70/15/15`
  - AdamW optimizer
  - ReduceLROnPlateau scheduler
  - early stopping patience `4`
- Added the second result plot:
  - `predictions_vs_actuals_<site>.png`
- Kept the existing loss plot:
  - `loss_curves_<site>.png`
- Added test result CSV output:
  - `test_results_<site>.csv`

## Validation Run

The requested compile check passed:

```bash
python3 -m py_compile app.py train_demo.py data_acquisition.py
```

Additional smoke checks were run with the virtual environment:

```bash
.venv/bin/python - <<'PY'
import app
import train_demo
import data_acquisition
print('imports ok')
PY
```

The import smoke test passed. Matplotlib printed cache/fontconfig warnings because some local cache directories were not writable, but imports completed successfully.

## GitHub Push

The latest changes were committed and pushed to GitHub.

Repository:

```text
https://github.com/yashsanap14/Water-prediction-colab-version
```

Latest pushed commit:

```text
5e4de40 Add ROI training modes and result plots
```

Remote branch:

```text
origin/main
```

## How To Run On Google Colab

First, set Colab runtime:

```text
Runtime > Change runtime type > T4 GPU
```

Then run these cells.

### Cell 1: Check GPU

```python
!nvidia-smi
```

### Cell 2: Clone Latest Repo And Install Dependencies

```python
%cd /content
!rm -rf Water-prediction-colab-version
!git clone https://github.com/yashsanap14/Water-prediction-colab-version.git
%cd Water-prediction-colab-version
!pip install -r requirements.txt -q
```

### Optional: Mount Google Drive

Only run this if you want to copy results to Google Drive from the app.

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Cell 3: Launch Gradio

```python
import os, sys

os.environ["WATER_LEVEL_DEMO_BASE_DIR"] = os.path.join(os.getcwd(), "water_level_demo")
sys.path.insert(0, ".")

import app
app.launch_gradio(share=True)
```

Colab will print a public Gradio URL. Open that link to use the app.

## File Locations On Colab

With the commands above, downloaded images are saved here:

```text
/content/Water-prediction-colab-version/water_level_demo/data/images/
```

The generated labels CSV is saved here:

```text
/content/Water-prediction-colab-version/water_level_demo/data/labels.csv
```

Training results are saved here:

```text
/content/Water-prediction-colab-version/water_level_demo/results/
```

Typical result files:

```text
best_model_<site>.pth
scaler_<site>.pkl
test_results_<site>.csv
loss_curves_<site>.png
predictions_vs_actuals_<site>.png
config.json
```

If Google Drive is mounted and the app option `Copy results to Google Drive` is enabled, results are also copied here:

```text
/content/drive/MyDrive/water_level_demo/results/
```

