import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF

# =========================================================
# ⚡ FER+ CLASS NAMES (8 EMOTIONS)
# =========================================================
# CRITICAL: This order must match your folder names exactly!
# Check with: ls data/train/
CLASS_NAMES = ['neutral', 'happy', 'suprise', 'sad', 'angry', 'disgust', 'fear', 'contempt']

# Alternative mappings if your folders use different names:
# If folders are: angry, disgust, fear, happy, sad, surprise, neutral, contempt
# CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'contempt']

# If folders are: Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear, Contempt (capitalized)
# CLASS_NAMES = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']


# =========================================================
# ⚡ YOUR ORIGINAL AUGMENTATIONS (UNCHANGED)
# =========================================================

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        y_a, y_b = batch_y, batch_y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch_x, batch_y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch_x.size(0)
        index = torch.randperm(batch_size)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(batch_x.size(), lam)
        batch_x[:, :, bbx1:bbx2, bby1:bby2] = batch_x[index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_x.size(-1) * batch_x.size(-2)))
        y_a, y_b = batch_y, batch_y[index]
        
        return batch_x, y_a, y_b, lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class RandomFaceCrop:
    def __init__(self, scale=(0.8, 1.0)):
        self.scale = scale
    
    def __call__(self, img):
        width, height = img.size
        scale = random.uniform(*self.scale)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        
        return TF.crop(img, top, left, new_height, new_width)


class RandomOcclusion:
    def __init__(self, p=0.5, scale=(0.02, 0.1)):
        self.p = p
        self.scale = scale
    
    def __call__(self, img):
        if random.random() < self.p:
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            size = random.randint(int(h * self.scale[0]), int(h * self.scale[1]))
            x = random.randint(0, w - size)
            y = random.randint(0, h - size)
            
            color = random.randint(0, 255)
            img_array[y:y+size, x:x+size] = color
            
            return Image.fromarray(img_array)
        return img


# =========================================================
# ⚡ YOUR ORIGINAL TRANSFORMS (UNCHANGED)
# =========================================================

def get_advanced_transforms(phase='train', img_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    
    if phase == "train":
        return T.Compose([
            T.Resize((img_size, img_size)),
            RandomFaceCrop(scale=(0.85, 1.0)),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(20, fill=128),
            T.RandomAffine(0, translate=(0.15, 0.15), scale=(0.9, 1.1), fill=128),
            T.ColorJitter(0.4, 0.4, 0.3, 0.1),
            RandomOcclusion(0.3, (0.02, 0.08)),
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            normalize,
            T.RandomErasing(0.25)
        ])
    
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            normalize
        ])


def get_transforms(phase="train", img_size=224):
    return get_advanced_transforms(phase, img_size)


# =========================================================
# ✅ EXPW DATASET LOADER (KEPT FOR COMPATIBILITY)
# =========================================================

class ExpWDataset(Dataset):
    """
    ExpW dataset format - keeping for compatibility
    """

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.img_dir = os.path.join(root_dir, "origin")
        self.transform = transform

        label_file = os.path.join(root_dir, "label.lst")
        self.samples = []

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()

                img_name = parts[0]
                expr_label = int(parts[-1])

                full_path = os.path.join(self.img_dir, img_name)

                if os.path.exists(full_path):
                    self.samples.append((full_path, expr_label))

        print(f"✅ Loaded ExpW: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================================================
# ⚡ FERImageFolder - WORKS FOR FER+! (UNCHANGED)
# =========================================================

class FERImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, classes=CLASS_NAMES, balance_classes=False):
        self.root = root_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.samples = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"Warning: {cls_dir} missing")
                continue
            
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

        print(f"Loaded {len(self.samples)} samples from {root_dir}")

        if balance_classes:
            self._balance()

    def _balance(self):
        from collections import defaultdict
        
        class_groups = defaultdict(list)
        for s in self.samples:
            class_groups[s[1]].append(s)

        max_len = max(len(v) for v in class_groups.values())
        new_samples = []

        for cls, items in class_groups.items():
            repeat = max_len // len(items)
            remainder = max_len % len(items)
            new_samples.extend(items * repeat)
            new_samples.extend(items[:remainder])

        self.samples = new_samples
        print(f"Balanced dataset to {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        return img, label


class TTADataset(Dataset):
    def __init__(self, base_dataset, n_tta=5):
        self.base_dataset = base_dataset
        self.n_tta = n_tta
        self.tta_transform = get_transforms("val", 224)

    def __len__(self):
        return len(self.base_dataset) * self.n_tta

    def __getitem__(self, idx):
        base_idx = idx // self.n_tta
        img, label = self.base_dataset[base_idx]

        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        img = self.tta_transform(img)
        return img, label, base_idx


# Add to emotion_backend/src/dataset.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms_advanced(phase='train', img_size=224):
    """
    Advanced augmentation using Albumentations
    Install: pip install albumentations
    """
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])