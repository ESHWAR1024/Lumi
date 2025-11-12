import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_resnet(num_classes=7, pretrained=True):
    """
    Creates a ResNet50 model for emotion classification.
    
    Args:
        num_classes: Number of emotion classes (default: 7)
        pretrained: Use ImageNet pretrained weights (default: True)
    
    Returns:
        ResNet50 model with custom classifier head
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze backbone layers if using pretrained weights
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def get_efficientnet(num_classes=7, pretrained=True):
    """
    Creates an EfficientNet-B0 model for emotion classification.
    
    Args:
        num_classes: Number of emotion classes (default: 7)
        pretrained: Use ImageNet pretrained weights (default: True)
    
    Returns:
        EfficientNet-B0 model with custom classifier head
    """
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Freeze backbone layers if using pretrained weights
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier
    # EfficientNet classifier is a Sequential with [Dropout, Linear]
    num_ftrs = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model