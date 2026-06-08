# AGENTS.md — Water Level ML Project Instructions

## Project goal
This project is a Google Colab + Gradio demo for USGS water-level prediction from HiVIS camera images. The production pipeline uses a large EfficientNet-L2 model, but the Colab demo should use smaller EfficientNet models such as B0, B3, or B4 that fit on a free Colab T4 GPU.

## Core ML workflow
The pipeline has four main steps:
1. Acquire USGS/NIMS images and NWIS water-level labels.
2. Select and verify ROI crop around the water gauge/waterline.
3. Train an EfficientNet regression model.
4. Run inference and generate evaluation plots.

## Important model rules
- This is a regression problem, not classification.
- The target is water level, usually `water_level` or USGS parameter `62620` for Pinewood/tidal sites.
- Do not train on precipitation `00045` as the target.
- Always save and reuse the target scaler.
- Inference must load the same scaler used during training.
- Inference must inverse-transform predictions before saving results.
- Training and inference must use the same:
  - model backbone
  - input image size
  - ROI crop
  - scaler
  - checkpoint/config

## EfficientNet guidance
Preferred Colab T4 options:
- `tf_efficientnet_b0.ns_jft_in1k`: fastest fallback
- `tf_efficientnet_b3.ns_jft_in1k`: default recommended demo model
- `tf_efficientnet_b4.ns_jft_in1k`: experimental, use smaller batch size

Avoid for Colab free tier:
- EfficientNet-L2 unless explicitly requested

Recommended default training config:
- backbone: `tf_efficientnet_b3.ns_jft_in1k`
- image size: 384
- batch size: 8
- learning rate: 0.0001
- epochs: 12
- validation split: 0.15
- test split: 0.15
- freeze ratio: 0.7
- optimizer: AdamW
- scheduler: ReduceLROnPlateau
- early stopping patience: 4

## ROI rules
The Colab/Gradio demo must use ROI format:
`(x1, y1, x2, y2)`

This matches PIL crop format:
`image.crop((x1, y1, x2, y2))`

For tensor images shaped `[C, H, W]`, crop using:
`image[:, y1:y2, x1:x2]`

Do not mix this with the old production format:
`((y1, x1), (y2, x2))`

Before training, log:
- ROI selected in UI
- ROI passed to train_model
- ROI received by Dataset
- crop width
- crop height
- crop orientation

Save 3–5 debug cropped images before resize/normalization to:
`/content/water_level_demo/results/training/debug_crops/`

## Output folder rules
Training outputs must be saved under:
`/content/water_level_demo/results/training/`

Inference outputs must be saved under:
`/content/water_level_demo/results/inference/`

Do not save duplicate generic files like:
- `best_model.pth`
- `scaler.pkl`
- `config.json`

Use site-specific names only, such as:
- `best_model_va_little_neck_creek_pinewood_rd_va_beach.pth`
- `scaler_va_little_neck_creek_pinewood_rd_va_beach.pkl`
- `config_va_little_neck_creek_pinewood_rd_va_beach.json`

## Required plots
Training/results should support:
1. Training loss plot
2. Predicted vs USGS observed regression plot with R², MAE, RMSE
3. Chronological ML predicted vs USGS observed time series plot

## Debugging rules
When modifying inference or training, add clear logs for:
- model path
- scaler path
- config path
- labels CSV path
- output directory
- backbone
- input image size
- ROI
- whether predictions are inverse-scaled

In Gradio/Colab, errors must be visible with:
- `debug=True`
- `show_error=True`
- `traceback.format_exc()`
- error logs saved to results folder

## Do not change unless requested
Do not change:
- model architecture
- scaler logic
- ROI convention
- train/validation/test split logic
- inference math
- plot definitions

unless the task specifically requires it.
