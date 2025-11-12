"""
Split FER2013 dataset from train/ into train/ and val/
Maintains class distribution and creates proper folder structure.

Usage:
    python split_dataset.py
    
This will:
- Keep 85% of train images in train/
- Move 15% of train images to val/
- Keep test/ folder untouched
- Maintain balanced class distribution
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Configuration
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_SPLIT = 0.15  # 15% for validation
RANDOM_SEED = 42

# Emotion classes
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def create_val_directory():
    """Create validation directory structure"""
    print("📁 Creating validation directory structure...")
    for emotion in EMOTIONS:
        val_emotion_dir = os.path.join(VAL_DIR, emotion)
        os.makedirs(val_emotion_dir, exist_ok=True)
    print("✅ Validation directories created\n")

def count_images(directory):
    """Count images in each emotion folder"""
    counts = {}
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(directory, emotion)
        if os.path.exists(emotion_dir):
            images = [f for f in os.listdir(emotion_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts[emotion] = len(images)
        else:
            counts[emotion] = 0
    return counts

def split_emotion_folder(emotion):
    """Split images from one emotion folder into train and val"""
    train_emotion_dir = os.path.join(TRAIN_DIR, emotion)
    val_emotion_dir = os.path.join(VAL_DIR, emotion)
    
    # Get all image files
    all_images = [f for f in os.listdir(train_emotion_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle with fixed seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)
    
    # Calculate split point
    total_images = len(all_images)
    val_count = int(total_images * VAL_SPLIT)
    
    # Split into validation and training
    val_images = all_images[:val_count]
    
    # Move validation images
    moved_count = 0
    for img in val_images:
        src = os.path.join(train_emotion_dir, img)
        dst = os.path.join(val_emotion_dir, img)
        shutil.move(src, dst)
        moved_count += 1
    
    return total_images, moved_count, total_images - moved_count

def main():
    print("=" * 60)
    print("🚀 FER2013 Dataset Splitter")
    print("=" * 60)
    print()
    
    # Check if train directory exists
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: {TRAIN_DIR} not found!")
        print("Please ensure your dataset is in the correct location.")
        return
    
    # Show current distribution
    print("📊 Current Training Set Distribution:")
    train_counts = count_images(TRAIN_DIR)
    total_train = sum(train_counts.values())
    
    for emotion, count in train_counts.items():
        print(f"   {emotion:10s}: {count:5d} images")
    print(f"   {'TOTAL':10s}: {total_train:5d} images\n")
    
    # Confirm split
    print(f"🔄 Split Configuration:")
    print(f"   Training:   {int((1-VAL_SPLIT)*100)}%")
    print(f"   Validation: {int(VAL_SPLIT*100)}%")
    print(f"   Random Seed: {RANDOM_SEED}\n")
    
    input("Press Enter to continue with the split...")
    print()
    
    # Create validation directories
    create_val_directory()
    
    # Split each emotion folder
    print("🔄 Splitting dataset...")
    results = {}
    
    for emotion in EMOTIONS:
        total, val_moved, train_remaining = split_emotion_folder(emotion)
        results[emotion] = {
            'total': total,
            'val': val_moved,
            'train': train_remaining
        }
        print(f"   {emotion:10s}: {train_remaining:5d} train | {val_moved:5d} val")
    
    print()
    
    # Final summary
    print("=" * 60)
    print("✅ Split Complete!")
    print("=" * 60)
    print()
    
    print("📊 Final Distribution:")
    print()
    print("TRAINING SET:")
    final_train_counts = count_images(TRAIN_DIR)
    for emotion, count in final_train_counts.items():
        percentage = (count / results[emotion]['total']) * 100
        print(f"   {emotion:10s}: {count:5d} images ({percentage:.1f}%)")
    print(f"   {'TOTAL':10s}: {sum(final_train_counts.values()):5d} images")
    
    print()
    print("VALIDATION SET:")
    final_val_counts = count_images(VAL_DIR)
    for emotion, count in final_val_counts.items():
        percentage = (count / results[emotion]['total']) * 100
        print(f"   {emotion:10s}: {count:5d} images ({percentage:.1f}%)")
    print(f"   {'TOTAL':10s}: {sum(final_val_counts.values()):5d} images")
    
    print()
    print("=" * 60)
    print("🎉 Dataset is ready for training!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run: .\\run_train.ps1")
    print("2. Or: python src/train.py --train_dir data/train --val_dir data/val")
    print()

if __name__ == "__main__":
    main()