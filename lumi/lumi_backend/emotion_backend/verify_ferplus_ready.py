"""
FER+ Training Readiness Checker
Verifies everything is configured correctly before training

Usage:
    python verify_ferplus_ready.py
"""

import os
import sys
from pathlib import Path
from collections import Counter

def check_dataset_structure():
    """Check if dataset folders exist and match FER+"""
    print("=" * 70)
    print("1️⃣  CHECKING DATASET STRUCTURE")
    print("=" * 70)
    
    data_dir = Path("data")
    required_splits = ["train", "val"]
    optional_splits = ["test"]
    
    issues = []
    warnings = []
    
    # Check data folder exists
    if not data_dir.exists():
        print("❌ CRITICAL: 'data' folder not found!")
        return False
    
    print("✅ Found 'data' folder")
    
    # Check train/val folders
    for split in required_splits:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"❌ CRITICAL: '{split}' folder not found!")
            issues.append(f"Missing {split} folder")
        else:
            print(f"✅ Found '{split}' folder")
    
    # Check test folder (optional)
    for split in optional_splits:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"⚠️  WARNING: '{split}' folder not found (optional)")
            warnings.append(f"Missing {split} folder")
        else:
            print(f"✅ Found '{split}' folder")
    
    if issues:
        return False
    
    return True


def check_emotion_folders():
    """Check if emotion folders exist and are populated"""
    print("\n" + "=" * 70)
    print("2️⃣  CHECKING EMOTION FOLDERS")
    print("=" * 70)
    
    # Possible emotion folder names
    fer_plus_emotions = ['neutral', 'happy', 'suprise', 'sad', 
                         'angry', 'disgust', 'fear', 'contempt']
    
    fer2013_emotions = ['angry', 'disgust', 'fear', 'happy', 
                        'neutral', 'sad', 'surprise']
    
    data_dir = Path("data")
    splits = ["train", "val", "test"]
    
    found_emotions = set()
    emotion_counts = {}
    
    for split in splits:
        split_path = data_dir / split
        if not split_path.exists():
            continue
        
        print(f"\n📁 {split}/")
        
        for emotion in fer_plus_emotions:
            emotion_path = split_path / emotion
            if emotion_path.exists():
                image_count = len(list(emotion_path.glob("*.png"))) + \
                             len(list(emotion_path.glob("*.jpg"))) + \
                             len(list(emotion_path.glob("*.jpeg")))
                
                if image_count > 0:
                    found_emotions.add(emotion)
                    emotion_counts[f"{split}/{emotion}"] = image_count
                    print(f"   ✅ {emotion:12s}: {image_count:5d} images")
                else:
                    print(f"   ⚠️  {emotion:12s}: EMPTY")
            else:
                print(f"   ❌ {emotion:12s}: NOT FOUND")
    
    # Check if using FER2013 names instead
    if not found_emotions:
        print("\n⚠️  FER+ emotion folders not found. Checking FER2013 format...")
        for split in splits:
            split_path = data_dir / split
            if not split_path.exists():
                continue
            
            for emotion in fer2013_emotions:
                emotion_path = split_path / emotion
                if emotion_path.exists():
                    image_count = len(list(emotion_path.glob("*.png"))) + \
                                 len(list(emotion_path.glob("*.jpg")))
                    if image_count > 0:
                        found_emotions.add(emotion)
                        print(f"   Found: {emotion} ({image_count} images)")
    
    if not found_emotions:
        print("\n❌ CRITICAL: No emotion folders found!")
        print("\nExpected FER+ structure:")
        print("data/")
        print("├── train/")
        print("│   ├── neutral/")
        print("│   ├── happiness/")
        print("│   ├── surprise/")
        print("│   ├── sadness/")
        print("│   ├── anger/")
        print("│   ├── disgust/")
        print("│   ├── fear/")
        print("│   └── contempt/")
        return False, None
    
    # Determine dataset type
    is_ferplus = any(e in fer_plus_emotions for e in found_emotions)
    is_fer2013 = any(e in fer2013_emotions for e in found_emotions)
    
    if is_ferplus:
        print(f"\n✅ Detected FER+ format ({len(found_emotions)} emotions)")
        expected = fer_plus_emotions
    elif is_fer2013:
        print(f"\n⚠️  Detected FER2013 format ({len(found_emotions)} emotions)")
        print("   FER+ has 8 classes, FER2013 has 7 classes")
        expected = fer2013_emotions
    else:
        print(f"\n⚠️  Unknown format ({len(found_emotions)} emotions)")
        expected = list(found_emotions)
    
    return True, expected


def check_class_names():
    """Check if CLASS_NAMES in dataset.py is correct"""
    print("\n" + "=" * 70)
    print("3️⃣  CHECKING CLASS_NAMES IN CODE")
    print("=" * 70)
    
    dataset_file = Path("src/dataset.py")
    
    if not dataset_file.exists():
        print("❌ CRITICAL: src/dataset.py not found!")
        return False, None
    
    print("✅ Found src/dataset.py")
    
    # Read CLASS_NAMES from file
    with open(dataset_file, 'r') as f:
        content = f.read()
        
        # Find CLASS_NAMES definition
        for line in content.split('\n'):
            if 'CLASS_NAMES' in line and '=' in line and not line.strip().startswith('#'):
                print(f"\n📝 Current CLASS_NAMES:")
                print(f"   {line.strip()}")
                
                # Extract class names
                if '[' in line and ']' in line:
                    import re
                    matches = re.findall(r"'([^']+)'", line)
                    return True, matches
    
    print("⚠️  Could not find CLASS_NAMES definition")
    return False, None


def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "=" * 70)
    print("4️⃣  CHECKING DEPENDENCIES")
    print("=" * 70)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn'
    }
    
    optional = {
        'tensorboard': 'TensorBoard (for monitoring)',
        'albumentations': 'Albumentations (advanced augmentation)'
    }
    
    missing = []
    missing_optional = []
    
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(name)
    
    for package, name in optional.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - not installed (optional)")
            missing_optional.append(name)
    
    if missing:
        print(f"\n❌ Missing required packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n💡 Optional packages not installed: {', '.join(missing_optional)}")
    
    return True


def check_gpu():
    """Check GPU availability"""
    print("\n" + "=" * 70)
    print("5️⃣  CHECKING GPU")
    print("=" * 70)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  No GPU detected - training will use CPU (SLOW)")
            print("   Consider using Google Colab or a GPU instance")
            return False
    except:
        print("❌ Could not check GPU status")
        return False


def main():
    print("=" * 70)
    print("🔍 FER+ TRAINING READINESS CHECK")
    print("=" * 70)
    print()
    
    # Run checks
    checks_passed = []
    
    # 1. Dataset structure
    if check_dataset_structure():
        checks_passed.append("Dataset structure")
    else:
        print("\n❌ Dataset structure check FAILED")
        return
    
    # 2. Emotion folders
    folders_ok, detected_emotions = check_emotion_folders()
    if folders_ok:
        checks_passed.append("Emotion folders")
    else:
        print("\n❌ Emotion folders check FAILED")
        return
    
    # 3. CLASS_NAMES
    classnames_ok, code_emotions = check_class_names()
    if classnames_ok:
        checks_passed.append("CLASS_NAMES")
        
        # Compare detected emotions with code
        if detected_emotions and code_emotions:
            if set(detected_emotions) == set(code_emotions):
                print(f"\n✅ CLASS_NAMES matches dataset: {len(code_emotions)} classes")
            else:
                print(f"\n⚠️  WARNING: CLASS_NAMES mismatch!")
                print(f"   In folders: {detected_emotions}")
                print(f"   In code: {code_emotions}")
                print(f"\n   You need to update src/dataset.py line with CLASS_NAMES")
    
    # 4. Dependencies
    if check_dependencies():
        checks_passed.append("Dependencies")
    else:
        print("\n❌ Dependencies check FAILED")
        return
    
    # 5. GPU
    has_gpu = check_gpu()
    if has_gpu:
        checks_passed.append("GPU")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Checks passed: {len(checks_passed)}/5")
    for check in checks_passed:
        print(f"   ✓ {check}")
    
    if len(checks_passed) >= 4:  # GPU is optional
        print("\n" + "=" * 70)
        print("🚀 YOU'RE READY TO TRAIN!")
        print("=" * 70)
        
        print("\n📝 TRAINING COMMANDS:")
        print("\n1️⃣  Quick test (30 epochs):")
        print("   python train_advanced_fixed.py \\")
        print("       --train_dir data/train \\")
        print("       --val_dir data/val \\")
        print("       --model efficientnetv2 \\")
        print("       --epochs 30 \\")
        print("       --batch_size 48 \\")
        print("       --use_amp")
        
        print("\n2️⃣  Full training (80 epochs, best accuracy):")
        print("   python train_advanced_fixed.py \\")
        print("       --train_dir data/train \\")
        print("       --val_dir data/val \\")
        print("       --model efficientnetv2 \\")
        print("       --epochs 80 \\")
        print("       --batch_size 48 \\")
        print("       --lr 1e-3 \\")
        print("       --img_size 224 \\")
        print("       --pretrained \\")
        print("       --use_amp \\")
        print("       --use_class_weights \\")
        print("       --loss focal \\")
        print("       --warmup_epochs 5 \\")
        print("       --patience 20 \\")
        print("       --save_dir models/ferplus_checkpoints \\")
        print("       --log_dir logs/ferplus")
        
        print("\n3️⃣  Monitor training:")
        print("   tensorboard --logdir logs/ferplus")
        
    else:
        print("\n" + "=" * 70)
        print("❌ NOT READY - FIX ISSUES ABOVE")
        print("=" * 70)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()