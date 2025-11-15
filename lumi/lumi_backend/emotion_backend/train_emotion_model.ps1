# PowerShell script to train advanced emotion model
# Place this file in: emotion_backend/train_emotion_model.ps1
# Usage: .\train_emotion_model.ps1

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🚀 Advanced Emotion Recognition Training" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

# Navigate to project directory
Set-Location $PSScriptRoot

# Activate virtual environment
Write-Host "📦 Activating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv\Scripts\Activate.ps1") {
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "❌ Virtual environment not found! Creating..." -ForegroundColor Red
    python -m venv venv
    .\venv\Scripts\Activate.ps1
}

# Install/upgrade dependencies
Write-Host "`n📦 Installing dependencies..." -ForegroundColor Cyan
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Check if data is prepared
if (-not (Test-Path "data\train")) {
    Write-Host "`n⚠️ Dataset not prepared!" -ForegroundColor Yellow
    Write-Host "Running dataset preparation..." -ForegroundColor Cyan
    python prepare_affectnet_dataset.py
}

# Training configuration
Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host "🎯 Training Configuration" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 69) -ForegroundColor Cyan
Write-Host ""

$MODEL = "efficientnetv2"  # Options: efficientnetv2, efficientnetb3, resnetv2
$EPOCHS = 100
$BATCH_SIZE = 64
$LR = 0.001
$IMG_SIZE = 224

Write-Host "Model:        $MODEL" -ForegroundColor White
Write-Host "Epochs:       $EPOCHS" -ForegroundColor White
Write-Host "Batch Size:   $BATCH_SIZE" -ForegroundColor White
Write-Host "Learning Rate: $LR" -ForegroundColor White
Write-Host "Image Size:   ${IMG_SIZE}x${IMG_SIZE}" -ForegroundColor White
Write-Host ""

# Confirm
$confirm = Read-Host "Press Enter to start training (or Ctrl+C to cancel)"

# Start training
Write-Host "`n🏋️ Starting training..." -ForegroundColor Green
Write-Host ""

python src/train_advanced.py `
    --model $MODEL `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --lr $LR `
    --img_size $IMG_SIZE `
    --pretrained `
    --use_amp `
    --use_class_weights `
    --loss focal `
    --warmup_epochs 5 `
    --patience 20 `
    --num_workers 4 `
    --save_dir models/checkpoints_advanced `
    --log_dir logs/advanced

# Check if training completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 69) -ForegroundColor Green
    Write-Host "✅ Training completed successfully!" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Green
    Write-Host ("=" * 69) -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 View training logs:" -ForegroundColor Cyan
    Write-Host "   tensorboard --logdir logs/advanced" -ForegroundColor White
    Write-Host ""
    Write-Host "🔍 Test your model:" -ForegroundColor Cyan
    Write-Host "   python realtime_inference.py --source webcam --arch $MODEL --model_path models/checkpoints_advanced/best_model.pt" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "`n❌ Training failed with error code: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")