import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_NAMES = ['angry','disgust','fear','happy','sad','surprise','neutral']

def get_transforms(phase='train', img_size=48):
    if phase == 'train':
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485], std=[0.229]) if False else T.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485,0.485,0.485],[0.229,0.229,0.229])
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
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
