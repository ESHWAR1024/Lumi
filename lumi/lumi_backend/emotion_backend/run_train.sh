#!/bin/bash

# Training script for emotion recognition model
# Usage: bash run_train.sh

set -e

echo "================================"
echo "Emotion Recognition Training"
echo "================================"

# Check if data directories exist
if [ ! -d "data/train" ]; then
    echo "✗ Error: data/train directory not found!"
    exit 1
fi

if [ ! -d "data/val" ]; then
    echo "✗ Error: data/val directory not found!"
    exit 1
fi

# Create necessary directories
mkdir -p models/checkpoints
mkdir -p logs

echo "✓ Data directories verified"
echo "✓ Output directories created"

# Create a Python script to run training
python3 << 'EOF'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import Trainer
from src.utils import get_device, setup_seed

# Set random seed for reproducibility
setup_seed(42)

# Get device
device = get_device()

# Initialize trainer
print("\nInitializing model...")
trainer = Trainer(model_name='resnet50', num_classes=7, device=str(device))

# Start training
print("\nStarting training...\n")
history = trainer.train(
    train_data_dir='./data/train',
    val_data_dir='./data/val',
    epochs=50,
    batch_size=32,
    lr=0.001,
    checkpoint_dir='./models/checkpoints',
    log_dir='./logs'
)

print("\n✓ Training completed successfully!")
print("✓ Best model saved to: models/checkpoints/best_model.pt")
print("✓ Training history saved to: logs/training_history.json")

EOF

echo ""
echo "================================"
echo "Training script finished"
echo "================================"
