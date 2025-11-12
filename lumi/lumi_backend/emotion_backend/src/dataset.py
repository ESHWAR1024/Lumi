import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def get_transforms(phase='train', img_size=224):
    """
    Get image transforms for training/validation.
    
    CRITICAL FIXES:
    1. Increased default image size to 224x224 (standard for ImageNet pretrained models)
    2. Fixed normalization to match ImageNet pretraining
    3. Added more aggressive augmentation for training
    """
    
    # ImageNet normalization (IMPORTANT for pretrained models!)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if phase == 'train':
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),  # Increased from 10
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # More aggressive
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add slight translation
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


def get_transforms_lightweight(phase='train', img_size=48):
    """
    Lighter transforms for 48x48 images (if using SimpleCNN).
    Uses simpler normalization since no pretrained weights.
    """
    if phase == 'train':
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Simple normalization
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


class FERImageFolder(Dataset):
    """
    Simple dataset expecting structure:
    root/class_x/xxx.png
    root/class_y/123.png
    """
    def __init__(self, root_dir, transform=None, classes=CLASS_NAMES):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.classes = classes
        
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label