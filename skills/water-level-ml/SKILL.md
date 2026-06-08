# Water-Level ML Skill

Use this skill when editing the water-level training, inference, ROI, or plotting pipeline.

## Checklist before changing code
- Confirm whether the task affects training, inference, ROI, plotting, or file paths.
- Check current output paths before changing them.
- Preserve Colab compatibility.
- Preserve Gradio UI behavior.
- Keep training and inference config consistent.

## Training checklist
- Confirm correct target column: `water_level` or `62620`.
- Confirm scaler is fit only on training targets.
- Save scaler with the model artifacts.
- Save config with backbone, image size, ROI, splits, learning rate, batch size, and seed.
- Save training artifacts only under `results/training/`.

## Inference checklist
- Load model from `results/training/`.
- Load scaler from same training run.
- Load ROI and image size from config.
- Apply same preprocessing as training.
- Inverse-transform predictions.
- Save outputs under `results/inference/`.

## ROI checklist
- Use `(x1, y1, x2, y2)`.
- Preview crop and training crop must match.
- Save debug crops before resize and normalization.
