"""
Prepare AffectNet dataset from label.lst format
Place this file in: emotion_backend/prepare_affectnet_dataset.py

Usage:
    python prepare_affectnet_dataset.py
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

# Configuration
IMAGES_DIR = "data/images"
LABEL_FILE = "data/label.lst"
OUTPUT_DIR = "data"
TRAIN_SPLIT = 0.80
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10
RANDOM_SEED = 42

# Emotion mapping
EMOTION_MAP = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral"
}

def parse_label_file(label_path):
    """
    Parse label.lst file.
    Format: image_name face_id top left right bottom confidence expression_label
    """
    print(f"📖 Reading labels from: {label_path}")
    
    samples = []
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 8:
                print(f"⚠️ Skipping malformed line {line_num}: {line}")
                continue
            
            image_name = parts[0]
            face_id = parts[1]
            box_top = float(parts[2])
            box_left = float(parts[3])
            box_right = float(parts[4])
            box_bottom = float(parts[5])
            confidence = float(parts[6])
            emotion_label = parts[7]
            
            # Skip low confidence detections
            if confidence < 0.8:
                continue
            
            # Map emotion label
            if emotion_label not in EMOTION_MAP:
                continue
            
            emotion_name = EMOTION_MAP[emotion_label]
            
            # Store sample info
            samples.append({
                'image_name': image_name,
                'face_id': face_id,
                'bbox': [box_top, box_left, box_right, box_bottom],
                'confidence': confidence,
                'emotion': emotion_name,
                'emotion_id': int(emotion_label)
            })
    
    print(f"✅ Loaded {len(samples)} samples")
    return samples

def print_distribution(samples, title):
    """Print emotion distribution"""
    from collections import Counter
    emotion_counts = Counter([s['emotion'] for s in samples])
    
    print(f"\n{title}")
    print("=" * 50)
    total = len(samples)
    for emotion in sorted(emotion_counts.keys()):
        count = emotion_counts[emotion]
        percentage = (count / total) * 100
        print(f"  {emotion:10s}: {count:6d} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total:6d}")
    print("=" * 50)

def stratified_split(samples, train_ratio, val_ratio, test_ratio, random_state=42):
    """Split dataset with stratification by emotion class"""
    # Group by emotion
    from collections import defaultdict
    emotion_groups = defaultdict(list)
    for sample in samples:
        emotion_groups[sample['emotion']].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # Split each emotion group
    for emotion, group_samples in emotion_groups.items():
        # Shuffle
        random.Random(random_state).shuffle(group_samples)
        
        n = len(group_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples.extend(group_samples[:train_end])
        val_samples.extend(group_samples[train_end:val_end])
        test_samples.extend(group_samples[val_end:])
    
    # Shuffle all splits
    random.Random(random_state).shuffle(train_samples)
    random.Random(random_state).shuffle(val_samples)
    random.Random(random_state).shuffle(test_samples)
    
    return train_samples, val_samples, test_samples

def crop_and_save_face(image_path, bbox, output_path, padding=0.2):
    """Crop face from image with padding"""
    from PIL import Image
    
    try:
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        
        top, left, right, bottom = bbox
        
        # Add padding
        box_width = right - left
        box_height = bottom - top
        
        left = max(0, left - box_width * padding)
        top = max(0, top - box_height * padding)
        right = min(width, right + box_width * padding)
        bottom = min(height, bottom + box_height * padding)
        
        # Crop
        face_img = img.crop((left, top, right, bottom))
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        face_img.save(output_path, quality=95)
        
        return True
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return False

def prepare_split(samples, split_name, images_dir, output_dir):
    """Prepare one data split (train/val/test)"""
    print(f"\n📦 Preparing {split_name} split...")
    
    success_count = 0
    fail_count = 0
    
    for sample in tqdm(samples, desc=f"Processing {split_name}"):
        # Source image path
        src_path = os.path.join(images_dir, sample['image_name'])
        
        if not os.path.exists(src_path):
            fail_count += 1
            continue
        
        # Destination path
        emotion = sample['emotion']
        face_id = sample['face_id']
        base_name = Path(sample['image_name']).stem
        dst_filename = f"{base_name}_face{face_id}.jpg"
        dst_path = os.path.join(output_dir, split_name, emotion, dst_filename)
        
        # Crop and save
        if crop_and_save_face(src_path, sample['bbox'], dst_path, padding=0.2):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"✅ {split_name}: {success_count} images processed, {fail_count} failed")

def main():
    print("=" * 60)
    print("🚀 AffectNet Dataset Preparation")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: Label file not found: {LABEL_FILE}")
        return
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Images directory not found: {IMAGES_DIR}")
        return
    
    # Parse labels
    samples = parse_label_file(LABEL_FILE)
    
    if len(samples) == 0:
        print("❌ No valid samples found!")
        return
    
    # Show distribution
    print_distribution(samples, "📊 Overall Dataset Distribution")
    
    # Split dataset
    print(f"\n🔀 Splitting dataset...")
    print(f"   Train: {TRAIN_SPLIT*100:.0f}%")
    print(f"   Val:   {VAL_SPLIT*100:.0f}%")
    print(f"   Test:  {TEST_SPLIT*100:.0f}%")
    
    train_samples, val_samples, test_samples = stratified_split(
        samples, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED
    )
    
    print_distribution(train_samples, "📊 Training Set Distribution")
    print_distribution(val_samples, "📊 Validation Set Distribution")
    print_distribution(test_samples, "📊 Test Set Distribution")
    
    # Prepare each split
    prepare_split(train_samples, 'train', IMAGES_DIR, OUTPUT_DIR)
    prepare_split(val_samples, 'val', IMAGES_DIR, OUTPUT_DIR)
    prepare_split(test_samples, 'test', IMAGES_DIR, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("🎉 Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the splits in data/train, data/val, data/test")
    print("2. Run training: python src/train_advanced.py")
    print("=" * 60)

if __name__ == "__main__":
    main()