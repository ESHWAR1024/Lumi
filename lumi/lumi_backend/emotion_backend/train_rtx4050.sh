#!/bin/bash

# ============================================================================
# RTX 4050 6GB Optimized Training Script
# Designed specifically for your hardware constraints
# ============================================================================

set -e

echo "========================================================================"
echo "🚀 RTX 4050 6GB OPTIMIZED EMOTION RECOGNITION TRAINING"
echo "========================================================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check CUDA availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Error checking CUDA. Make sure PyTorch is installed correctly.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ GPU check passed${NC}"
echo ""

# Check if data directories exist
if [ ! -d "data/train" ]; then
    echo -e "${RED}❌ Error: data/train directory not found!${NC}"
    echo "Please organize your dataset in the following structure:"
    echo "data/"
    echo "├── train/"
    echo "│   ├── angry/"
    echo "│   ├── disgust/"
    echo "│   ├── fear/"
    echo "│   ├── happy/"
    echo "│   ├── sad/"
    echo "│   ├── surprise/"
    echo "│   └── neutral/"
    echo "└── val/"
    echo "    └── [same structure]"
    exit 1
fi

if [ ! -d "data/val" ]; then
    echo -e "${RED}❌ Error: data/val directory not found!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Data directories found${NC}"
echo ""

# Create output directories
mkdir -p models/rtx4050_optimized
mkdir -p logs/rtx4050_optimized

echo "========================================================================"
echo "📊 TRAINING CONFIGURATION"
echo "========================================================================"
echo "Architecture: EfficientNetV2-S (Best for 6GB VRAM)"
echo "Batch Size: 48 (Optimized for your GPU)"
echo "Image Size: 224x224"
echo "Mixed Precision: Enabled (Saves 40% VRAM)"
echo "Epochs: 80 (with early stopping)"
echo "Augmentation: MixUp + CutMix + Advanced"
echo "Loss: Focal Loss (handles imbalance)"
echo "Optimizer: AdamW"
echo "Scheduler: Cosine Annealing"
echo "========================================================================"
echo ""

read -p "Press Enter to start training (or Ctrl+C to cancel)..."

echo ""
echo -e "${GREEN}🚀 Starting training...${NC}"
echo ""

# Run training with optimal settings for RTX 4050 6GB
python train_advanced.py \
  --train_dir data/train \
  --val_dir data/val \
  --arch efficientnetv2 \
  --epochs 80 \
  --batch_size 48 \
  --lr 3e-4 \
  --img_size 224 \
  --num_workers 4 \
  --pretrained \
  --use_mixup \
  --use_cutmix \
  --use_focal_loss \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --label_smoothing 0.1 \
  --scheduler cosine \
  --t0 15 \
  --patience 20 \
  --balance_classes \
  --save_dir models/rtx4050_optimized \
  --log_dir logs/rtx4050_optimized \
  --save_freq 10 \
  --seed 42

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ TRAINING COMPLETE!${NC}"
echo "========================================================================"
echo "📁 Models saved to: models/rtx4050_optimized/"
echo "📊 Logs saved to: logs/rtx4050_optimized/"
echo ""
echo "Next steps:"
echo "1. View training progress: tensorboard --logdir logs/rtx4050_optimized"
echo "2. Test the model: python test_model_advanced.py --test_dir data/test --model_path models/rtx4050_optimized/best_model.pt --arch efficientnetv2"
echo "3. Run webcam: python realtime_inference.py --source webcam --model_path models/rtx4050_optimized/best_model.pt --arch efficientnetv2"
echo "========================================================================"