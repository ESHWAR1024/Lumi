# ============================================================================
# RTX 4050 6GB Optimized Training Script (Windows PowerShell)
# Designed specifically for your hardware constraints
# ============================================================================

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "🚀 RTX 4050 6GB OPTIMIZED EMOTION RECOGNITION TRAINING" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

# Check CUDA availability
Write-Host "Checking GPU availability..." -ForegroundColor Yellow
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else `"None`"}')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error checking CUDA. Make sure PyTorch is installed correctly." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ GPU check passed" -ForegroundColor Green
Write-Host ""

# Check if data directories exist
if (-not (Test-Path "data/train")) {
    Write-Host "❌ Error: data/train directory not found!" -ForegroundColor Red
    Write-Host "Please organize your dataset in the following structure:"
    Write-Host "data/"
    Write-Host "├── train/"
    Write-Host "│   ├── angry/"
    Write-Host "│   ├── disgust/"
    Write-Host "│   ├── fear/"
    Write-Host "│   ├── happy/"
    Write-Host "│   ├── sad/"
    Write-Host "│   ├── surprise/"
    Write-Host "│   └── neutral/"
    Write-Host "└── val/"
    Write-Host "    └── [same structure]"
    exit 1
}

if (-not (Test-Path "data/val")) {
    Write-Host "❌ Error: data/val directory not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Data directories found" -ForegroundColor Green
Write-Host ""

# Create output directories
New-Item -ItemType Directory -Force -Path "models/rtx4050_optimized" | Out-Null
New-Item -ItemType Directory -Force -Path "logs/rtx4050_optimized" | Out-Null

Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "📊 TRAINING CONFIGURATION" -ForegroundColor Yellow
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "Architecture: EfficientNetV2-S (Best for 6GB VRAM)"
Write-Host "Batch Size: 48 (Optimized for your GPU)"
Write-Host "Image Size: 224x224"
Write-Host "Mixed Precision: Enabled (Saves 40% VRAM)"
Write-Host "Epochs: 80 (with early stopping)"
Write-Host "Augmentation: MixUp + CutMix + Advanced"
Write-Host "Loss: Focal Loss (handles imbalance)"
Write-Host "Optimizer: AdamW"
Write-Host "Scheduler: Cosine Annealing"
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host ""

$continue = Read-Host "Press Enter to start training (or Ctrl+C to cancel)"

Write-Host ""
Write-Host "🚀 Starting training..." -ForegroundColor Green
Write-Host ""

# Run training with optimal settings for RTX 4050 6GB
python train_advanced.py `
  --train_dir data/train `
  --val_dir data/val `
  --arch efficientnetv2 `
  --epochs 80 `
  --batch_size 48 `
  --lr 3e-4 `
  --img_size 224 `
  --num_workers 4 `
  --pretrained `
  --use_mixup `
  --use_cutmix `
  --use_focal_loss `
  --focal_alpha 0.25 `
  --focal_gamma 2.0 `
  --label_smoothing 0.1 `
  --scheduler cosine `
  --t0 15 `
  --patience 20 `
  --balance_classes `
  --save_dir models/rtx4050_optimized `
  --log_dir logs/rtx4050_optimized `
  --save_freq 10 `
  --seed 42

Write-Host ""
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "✅ TRAINING COMPLETE!" -ForegroundColor Green
Write-Host "========================================================================" -ForegroundColor Cyan
Write-Host "📁 Models saved to: models/rtx4050_optimized/"
Write-Host "📊 Logs saved to: logs/rtx4050_optimized/"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. View training progress: tensorboard --logdir logs/rtx4050_optimized"
Write-Host "2. Test the model: python test_model_advanced.py --test_dir data/test --model_path models/rtx4050_optimized/best_model.pt --arch efficientnetv2"
Write-Host "3. Run webcam: python realtime_inference.py --source webcam --model_path models/rtx4050_optimized/best_model.pt --arch efficientnetv2"
Write-Host "========================================================================" -ForegroundColor Cyan