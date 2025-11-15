"""
Advanced emotion recognition models with attention mechanisms
Place this file in: emotion_backend/src/model_advanced.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


# ======================================================================
# ATTENTION BLOCK
# ======================================================================
class AttentionBlock(nn.Module):
    """Channel + Spatial Attention"""
    def __init__(self, channels):
        super().__init__()

        # Channel attention (SE-style)
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
        # -------- Channel attention --------
        b, c, h, w = x.size()
        y = self.channel_pool(x).view(b, c)
        y = self.channel_fc(y).view(b, c, 1, 1)
        x = x * y

        # -------- Spatial attention --------
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_conv(sa)
        sa = self.spatial_sigmoid(sa)

        return x * sa



# ======================================================================
# EfficientNetV2
# ======================================================================
class EfficientNetV2Emotion(nn.Module):
    """
    EfficientNetV2-S + Attention
    Best accuracy on FER+/ExpW with 6GB VRAM
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # TIMM EfficientNetV2-S
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        # Get feature dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.backbone(dummy)
        self.feature_dim = feat.shape[1]

        # Attention
        self.attention = AttentionBlock(self.feature_dim)

        # Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.attention(feats)
        feats = self.global_pool(feats)
        feats = feats.flatten(1)
        return self.classifier(feats)



# ======================================================================
# EfficientNet-B3
# ======================================================================
class EfficientNetB3Emotion(nn.Module):
    """EfficientNet-B3 + Attention"""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )

        self.feature_dim = 1536  # fixed in B3 architecture
        self.attention = AttentionBlock(self.feature_dim)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.feature_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.attention(feats)
        feats = self.global_pool(feats).flatten(1)
        return self.classifier(feats)



# ======================================================================
# ResNet50 V2
# ======================================================================
class ResNetEmotionV2(nn.Module):
    """ResNet50 + Attention"""
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # Improved ResNet50 weights
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)

        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Attention on last feature map
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
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.attention(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)

        return self.classifier(x)



# ======================================================================
# MODEL FACTORY (THIS IS WHAT realtime_inference.py CALLS)
# ======================================================================
def get_model(model_name='efficientnetv2', num_classes=7, pretrained=True):
    """
    Factory loader for advanced models used in realtime_inference.py
    """

    model_name = model_name.lower()

    MODELS = {
        "efficientnetv2": EfficientNetV2Emotion,
        "efficientnetb3": EfficientNetB3Emotion,
        "resnetv2": ResNetEmotionV2,
    }

    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from {list(MODELS.keys())}")

    return MODELS[model_name](num_classes=num_classes, pretrained=pretrained)



# ======================================================================
# SELF-TEST (optional)
# ======================================================================
if __name__ == "__main__":
    print("Testing models...")

    for name in ["efficientnetv2", "efficientnetb3", "resnetv2"]:
        print(f"\n{name}:")
        model = get_model(name, num_classes=7, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        y = model(x)
        print("Output shape:", y.shape)

    print("\n✅ All advanced models OK!")
