import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm  # PyTorch Image Models - more architectures

class AttentionBlock(nn.Module):
    """Spatial attention mechanism to focus on important facial features"""
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales for better emotion recognition"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return self.relu(self.bn(out))


class EnhancedEmotionCNN(nn.Module):
    """
    State-of-the-art emotion recognition model with:
    - EfficientNetV2 backbone (better than V1)
    - Attention mechanisms
    - Multi-scale feature extraction
    - Ensemble-ready architecture
    """
    def __init__(self, num_classes=7, pretrained=True, dropout=0.3):
        super().__init__()
        
        # Use EfficientNetV2-S (balanced speed/accuracy)
        self.backbone = timm.create_model('efficientnetv2_rw_s', pretrained=pretrained, 
                                          num_classes=0, global_pool='')
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Attention mechanism
        self.attention = AttentionBlock(self.feature_dim)
        
        # Multi-scale feature extraction
        self.multi_scale = MultiScaleFeatureExtractor(self.feature_dim, self.feature_dim)
        
        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        
        # Classifier head with better regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Multi-scale features
        features = self.multi_scale(features)
        
        # Global pooling (both average and max)
        avg_pool = self.gap(features).flatten(1)
        max_pool = self.gmp(features).flatten(1)
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Classify
        out = self.classifier(features)
        return out


def get_efficientnetv2(num_classes=7, pretrained=True, model_size='s'):
    """
    Get EfficientNetV2 model (improved version)
    
    Args:
        num_classes: Number of emotion classes
        pretrained: Use ImageNet pretrained weights
        model_size: 's' (small), 'm' (medium), 'l' (large)
    """
    model_name = f'efficientnetv2_rw_{model_size}'
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_convnext(num_classes=7, pretrained=True, model_size='tiny'):
    """
    ConvNeXt - Modern CNN architecture (2022)
    Better than ResNet/EfficientNet in many tasks
    
    Args:
        model_size: 'tiny', 'small', 'base'
    """
    model_name = f'convnext_{model_size}'
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_swin_transformer(num_classes=7, pretrained=True, model_size='tiny'):
    """
    Swin Transformer - Vision Transformer with hierarchical features
    Excellent for facial expression recognition
    
    Args:
        model_size: 'tiny', 'small', 'base'
    """
    model_name = f'swin_{model_size}_patch4_window7_224'
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_ensemble_model(num_classes=7, pretrained=True):
    """
    Ensemble of multiple architectures for best accuracy
    Combines EfficientNetV2, ConvNeXt, and Swin Transformer
    """
    class EnsembleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model1 = get_efficientnetv2(num_classes, pretrained, 's')
            self.model2 = get_convnext(num_classes, pretrained, 'tiny')
            self.model3 = get_swin_transformer(num_classes, pretrained, 'tiny')
            
            # Learnable weights for ensemble
            self.weights = nn.Parameter(torch.ones(3) / 3)
        
        def forward(self, x):
            out1 = self.model1(x)
            out2 = self.model2(x)
            out3 = self.model3(x)
            
            # Weighted ensemble
            w = F.softmax(self.weights, dim=0)
            return w[0] * out1 + w[1] * out2 + w[2] * out3
    
    return EnsembleModel()


# Legacy support
def get_resnet(num_classes=7, pretrained=True, freeze_backbone=False):
    """ResNet50 - kept for backward compatibility"""
    model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    
    if freeze_backbone and pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


def get_resnet18(num_classes=7, pretrained=True):
    """ResNet18 - lighter version"""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return model


def get_efficientnet_b0(num_classes=7, pretrained=True):
    """EfficientNet-B0 - kept for backward compatibility"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    return model


class SimpleCNN(nn.Module):
    """Lightweight CNN for quick experiments"""
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
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x