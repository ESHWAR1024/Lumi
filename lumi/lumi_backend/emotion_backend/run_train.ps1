# --------------------------------------------
# PowerShell script to activate venv and train
# --------------------------------------------
Write-Host "🚀 Starting Emotion Model Training..."

# Move to project directory
Set-Location "C:\Users\ESHWAR\Documents\GitHub\Lumi\lumi\lumi_backend\emotion_backend"

# Activate virtual environment
Write-Host "🔹 Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Upgrade pip and install deps
Write-Host "🔹 Ensuring dependencies are up to date..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt

# Run training
Write-Host "🔹 Training started (using GPU if available)..."
python src/train.py --train_dir data/train --val_dir data/val --epochs 30 --batch_size 64 --arch resnet --pretrained --save_dir models/checkpoints

Write-Host "✅ Training complete! Check models/checkpoints/ for saved model."
