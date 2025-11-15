"""
Advanced emotion recognition models with attention mechanisms
Place this file in: emotion_backend/src/model_advanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class AttentionBlock(nn.Module):
    """Channel and Spatial Attention Module"""
    def __init__(self, channels):
        super().__init__()
        # Channel attention
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.spatial_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel attention
        b, c, h, w = x.size()
        y = self.channel_pool(x).view(b, c)
        y = self.channel_fc(y).view(b, c, 1, 1)
        x = x * y
        
        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        spatial = self.spatial_conv(spatial)
        spatial = self.spatial_sigmoid(spatial)
        x = x * spatial
        
        return x


class EfficientNetV2Emotion(nn.Module):
    """
    EfficientNetV2 with attention mechanisms
    Best for accuracy with 6GB VRAM
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # Load EfficientNetV2-S (smaller than B0 but more accurate)
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool=''  # Remove global pooling
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[1]
        
        # Add attention
        self.attention = AttentionBlock(self.feature_dim)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Global pooling
        features = self.global_pool(features)
        features = features.flatten(1)
        
        # Classify
        output = self.classifier(features)
        
        return output


class EfficientNetB3Emotion(nn.Module):
    """
    EfficientNet-B3 with attention (larger, more accurate)
    Still fits in 6GB VRAM with mixed precision
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        self.backbone = timm.create_model(
            'tf_efficientnet_b3',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        # Feature dimension for B3
        self.feature_dim = 1536
        
        # Attention
        self.attention = AttentionBlock(self.feature_dim)
        
        # Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.attention(features)
        features = self.global_pool(features).flatten(1)
        output = self.classifier(features)
        return output


class ResNetEmotionV2(nn.Module):
    """
    ResNet50 with modern improvements
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        
        # Use ResNet50 with better weights
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Remove final FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add attention after layer4
        self.attention = AttentionBlock(2048)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Forward through backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classify
        output = self.classifier(x)
        
        return output


def get_model(model_name='efficientnetv2', num_classes=7, pretrained=True):
    """
    Factory function to get model by name
    
    Args:
        model_name: 'efficientnetv2', 'efficientnetb3', 'resnetv2'
        num_classes: Number of emotion classes
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        Model instance
    """
    models_dict = {
        'efficientnetv2': EfficientNetV2Emotion,
        'efficientnetb3': EfficientNetB3Emotion,
        'resnetv2': ResNetEmotionV2
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models_dict.keys())}")
    
    model_class = models_dict[model_name]
    return model_class(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    for name in ['efficientnetv2', 'efficientnetb3', 'resnetv2']:
        print(f"\n{name}:")
        model = get_model(name, num_classes=7, pretrained=False)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        print(f"  Output shape: {y.shape}")
        
    print("\n✅ All models working!")