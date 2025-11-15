"""
Prepare ExpW dataset from single origin/ folder + label.lst
This will split your data into train/val/test with proper folder structure

Your current structure:
data/
├── origin/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── label.lst

After running this script:
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]

Usage:
    python prepare_expw_dataset.py
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from collections import defaultdict, Counter

# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS TO MATCH YOUR SETUP
# ============================================================================
IMAGES_DIR = "data/origin"           # Your images folder
LABEL_FILE = "data/label.lst"        # Your label file
OUTPUT_DIR = "data"                  # Output directory

# Split ratios
TRAIN_SPLIT = 0.70   # 70% for training
VAL_SPLIT = 0.15     # 15% for validation
TEST_SPLIT = 0.15    # 15% for testing

RANDOM_SEED = 42

# Emotion mapping (ExpW uses 0-6)
EMOTION_MAP = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral"
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def parse_label_file(label_path):
    """
    Parse label.lst file
    Format: image_name face_id top left right bottom confidence expression_label
    """
    print(f"📖 Reading labels from: {label_path}")
    
    samples = []
    skipped = 0
    
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines or comments
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            # Check if line has enough fields
            if len(parts) < 8:
                print(f"⚠️  Line {line_num}: Not enough fields, skipping")
                skipped += 1
                continue
            
            image_name = parts[0]
            face_id = parts[1]
            box_top = float(parts[2])
            box_left = float(parts[3])
            box_right = float(parts[4])
            box_bottom = float(parts[5])
            confidence = float(parts[6])
            emotion_label = parts[7]
            
            # Skip low confidence detections (optional)
            if confidence < 0.5:
                skipped += 1
                continue
            
            # Map emotion label
            if emotion_label not in EMOTION_MAP:
                print(f"⚠️  Line {line_num}: Unknown emotion {emotion_label}, skipping")
                skipped += 1
                continue
            
            emotion_name = EMOTION_MAP[emotion_label]
            
            # Check if image exists
            img_path = os.path.join(IMAGES_DIR, image_name)
            if not os.path.exists(img_path):
                skipped += 1
                continue
            
            # Store sample info
            samples.append({
                'image_name': image_name,
                'face_id': face_id,
                'bbox': [box_top, box_left, box_right, box_bottom],
                'confidence': confidence,
                'emotion': emotion_name,
                'emotion_id': int(emotion_label)
            })
    
    print(f"✅ Loaded {len(samples)} valid samples")
    if skipped > 0:
        print(f"⚠️  Skipped {skipped} samples (low confidence or missing)")
    
    return samples


def print_distribution(samples, title):
    """Print emotion distribution"""
    emotion_counts = Counter([s['emotion'] for s in samples])
    
    print(f"\n{title}")
    print("=" * 60)
    total = len(samples)
    for emotion in sorted(EMOTION_MAP.values()):
        count = emotion_counts.get(emotion, 0)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  {emotion:10s}: {count:6d} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':10s}: {total:6d}")
    print("=" * 60)


def stratified_split(samples, train_ratio, val_ratio, test_ratio, random_state=42):
    """Split dataset with stratification by emotion class"""
    # Group by emotion
    emotion_groups = defaultdict(list)
    for sample in samples:
        emotion_groups[sample['emotion']].append(sample)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    print(f"\n🔀 Splitting each emotion class...")
    
    # Split each emotion group
    for emotion, group_samples in emotion_groups.items():
        # Shuffle
        random.Random(random_state).shuffle(group_samples)
        
        n = len(group_samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train = group_samples[:train_end]
        val = group_samples[train_end:val_end]
        test = group_samples[val_end:]
        
        train_samples.extend(train)
        val_samples.extend(val)
        test_samples.extend(test)
        
        print(f"  {emotion:10s}: {len(train):4d} train | {len(val):4d} val | {len(test):4d} test")
    
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
        
        # Resize to 224x224 for consistency
        face_img = face_img.resize((224, 224), Image.LANCZOS)
        
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
        
        # Create unique filename
        dst_filename = f"{base_name}_face{face_id}.jpg"
        dst_path = os.path.join(output_dir, split_name, emotion, dst_filename)
        
        # Crop and save
        if crop_and_save_face(src_path, sample['bbox'], dst_path, padding=0.2):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"✅ {split_name}: {success_count} images saved, {fail_count} failed")
    return success_count, fail_count


def main():
    print("=" * 70)
    print("🚀 ExpW Dataset Preparation")
    print("=" * 70)
    
    # Check if files exist
    if not os.path.exists(LABEL_FILE):
        print(f"❌ Error: Label file not found: {LABEL_FILE}")
        print(f"   Please make sure label.lst is in the correct location")
        return
    
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ Error: Images directory not found: {IMAGES_DIR}")
        print(f"   Please make sure origin/ folder is in the correct location")
        return
    
    # Count images in origin folder
    image_files = [f for f in os.listdir(IMAGES_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Found {len(image_files)} images in {IMAGES_DIR}")
    
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
    
    print_distribution(train_samples, "\n📊 Training Set Distribution")
    print_distribution(val_samples, "\n📊 Validation Set Distribution")
    print_distribution(test_samples, "\n📊 Test Set Distribution")
    
    # Confirm before processing
    print("\n" + "=" * 70)
    print("⚠️  This will create cropped face images in:")
    print(f"   {os.path.join(OUTPUT_DIR, 'train/')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'val/')}")
    print(f"   {os.path.join(OUTPUT_DIR, 'test/')}")
    print("=" * 70)
    
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cancelled by user")
        return
    
    # Prepare each split
    train_success, train_fail = prepare_split(train_samples, 'train', IMAGES_DIR, OUTPUT_DIR)
    val_success, val_fail = prepare_split(val_samples, 'val', IMAGES_DIR, OUTPUT_DIR)
    test_success, test_fail = prepare_split(test_samples, 'test', IMAGES_DIR, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 Dataset Preparation Complete!")
    print("=" * 70)
    print(f"\n📊 Final Statistics:")
    print(f"   Training:   {train_success:5d} images ({train_fail} failed)")
    print(f"   Validation: {val_success:5d} images ({val_fail} failed)")
    print(f"   Test:       {test_success:5d} images ({test_fail} failed)")
    print(f"   TOTAL:      {train_success + val_success + test_success:5d} images")
    
    print("\n📁 Dataset structure created:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── train/")
    for emotion in sorted(EMOTION_MAP.values()):
        print(f"   │   ├── {emotion}/")
    print(f"   ├── val/")
    print(f"   │   └── [same structure]")
    print(f"   └── test/")
    print(f"       └── [same structure]")
    
    print("\n" + "=" * 70)
    print("🚀 Next Steps:")
    print("=" * 70)
    print("1. Verify the data:")
    print(f"   ls {OUTPUT_DIR}/train/")
    print(f"   ls {OUTPUT_DIR}/val/")
    print("\n2. Start training:")
    print("   python train_advanced_fixed.py --train_dir data/train --val_dir data/val")
    print("\n   Or with specific settings:")
    print("   python train_advanced_fixed.py \\")
    print("       --train_dir data/train \\")
    print("       --val_dir data/val \\")
    print("       --model efficientnetv2 \\")
    print("       --batch_size 48 \\")
    print("       --epochs 80 \\")
    print("       --use_amp")
    print("=" * 70)


if __name__ == "__main__":
    # Check if PIL is available
    try:
        from PIL import Image
    except ImportError:
        print("❌ Error: PIL (Pillow) is required")
        print("   Install it with: pip install Pillow")
        exit(1)
    
    main()