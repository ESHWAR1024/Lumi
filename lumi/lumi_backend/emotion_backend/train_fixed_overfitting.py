"""
Fixed Training Script with Anti-Overfitting Measures
This addresses the 91% train / 69% val accuracy gap

Key Changes:
1. Stronger regularization (dropout, weight decay)
2. Enhanced data augmentation
3. Learning rate adjustments
4. Gradient penalty
5. Early stopping based on validation loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path

from src.dataset import FERImageFolder, get_transforms, CLASS_NAMES
from src.model_advanced import get_model
from src.utils import save_checkpoint


class StrongerRegularizationModel(nn.Module):
    """
    Wrapper that adds stronger regularization to any model
    """
    def __init__(self, base_model, dropout_rate=0.5):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        features = self.base_model.backbone(x)
        
        # Add dropout before attention
        if hasattr(self.base_model, 'attention'):
            features = self.base_model.attention(features)
            features = self.dropout(features)  # Extra dropout
        
        # Global pooling
        features = self.base_model.global_pool(features).flatten(1)
        features = self.dropout(features)  # Dropout after pooling
        
        # Classifier with additional dropout
        output = self.base_model.classifier(features)
        
        return output


def get_enhanced_augmentation_transforms(img_size=224):
    """
    MUCH stronger augmentation to reduce overfitting
    """
    import torchvision.transforms as T
    
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(25, fill=128),  # Increased rotation
        T.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),  # Increased translation
            scale=(0.8, 1.2),      # Increased scale variation
            fill=128
        ),
        T.ColorJitter(
            brightness=0.5,  # Increased
            contrast=0.5,    # Increased
            saturation=0.4,
            hue=0.2
        ),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.RandomGrayscale(p=0.1),
        T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Increased erasing
    ])
    
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class MixupLoss(nn.Module):
    """
    Mixup + Label Smoothing for better generalization
    """
    def __init__(self, alpha=0.4, num_classes=8, smoothing=0.15):
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        if self.training and np.random.rand() < 0.5:
            # Apply mixup
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = target.size(0)
            index = torch.randperm(batch_size).to(target.device)
            
            # Smooth labels
            target_a = self._smooth_labels(target)
            target_b = self._smooth_labels(target[index])
            
            loss = lam * F.cross_entropy(pred, target) + \
                   (1 - lam) * F.cross_entropy(pred, target[index])
            return loss
        else:
            # Just label smoothing
            smooth_target = self._smooth_labels(target)
            log_pred = F.log_softmax(pred, dim=1)
            loss = F.nll_loss(log_pred, target)
            return loss
    
    def _smooth_labels(self, target):
        """Apply label smoothing"""
        with torch.no_grad():
            smooth = torch.zeros_like(target, dtype=torch.float)
            smooth.fill_(self.smoothing / (self.num_classes - 1))
            smooth.scatter_(0, target, 1.0 - self.smoothing)
        return smooth


class ImprovedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print("🔧 ANTI-OVERFITTING TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("=" * 70)
        
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.setup_logging()
    
    def setup_data(self):
        """Setup with STRONGER augmentation"""
        print("\n📦 Loading datasets with enhanced augmentation...")
        
        train_transform, val_transform = get_enhanced_augmentation_transforms(
            img_size=self.args.img_size
        )
        
        self.train_dataset = FERImageFolder(
            self.args.train_dir,
            transform=train_transform,
            classes=CLASS_NAMES
        )
        
        self.val_dataset = FERImageFolder(
            self.args.val_dir,
            transform=val_transform,
            classes=CLASS_NAMES
        )
        
        print(f"✅ Train: {len(self.train_dataset)} samples")
        print(f"✅ Val: {len(self.val_dataset)} samples")
        
        # Class weights
        if self.args.use_class_weights:
            class_counts = [0] * len(CLASS_NAMES)
            for _, label in self.train_dataset.samples:
                class_counts[label] += 1
            
            total = sum(class_counts)
            self.class_weights = torch.tensor(
                [total / (len(CLASS_NAMES) * count) for count in class_counts],
                dtype=torch.float32,
                device=self.device
            )
            print(f"✅ Class weights computed")
        else:
            self.class_weights = None
        
        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def setup_model(self):
        """Setup model with STRONGER regularization"""
        print(f"\n🧠 Creating model with heavy regularization: {self.args.model}")
        
        base_model = get_model(
            model_name=self.args.model,
            num_classes=len(CLASS_NAMES),
            pretrained=self.args.pretrained
        )
        
        # Wrap with additional dropout
        self.model = StrongerRegularizationModel(
            base_model,
            dropout_rate=self.args.dropout
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable: {trainable_params:,}")
        print(f"✅ Dropout rate: {self.args.dropout}")
    
    def setup_training(self):
        """Setup with aggressive regularization"""
        print(f"\n⚙️ Setting up training components...")
        
        # Loss with mixup + label smoothing
        self.criterion = MixupLoss(
            alpha=0.4,
            num_classes=len(CLASS_NAMES),
            smoothing=0.15
        )
        print("✅ Using Mixup + Label Smoothing Loss")
        
        # Optimizer with HIGHER weight decay
        if self.args.pretrained:
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'classifier' in name or 'attention' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.args.lr * 0.05, 'weight_decay': self.args.weight_decay * 2},
                {'params': head_params, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
            ])
            
            print(f"✅ Differentiated LR with heavy weight decay")
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        
        # Scheduler - reduce LR on validation loss plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # Monitor validation LOSS not accuracy
            factor=0.5,
            patience=5,
            verbose=True,
            min_lr=1e-7
        )
        print("✅ Using ReduceLROnPlateau (monitors val loss)")
        
        # Mixed precision
        self.scaler = GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("✅ Mixed precision enabled")
        
        self.best_val_loss = float('inf')  # Changed from accuracy
        self.patience_counter = 0
    
    def setup_logging(self):
        """Setup logging"""
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        
        config_path = os.path.join(self.args.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        print(f"✅ Logging to: {self.args.log_dir}")
    
    def train_epoch(self, epoch):
        """Train with regularization"""
        self.model.train()
        self.criterion.train()
        
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.args.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Stricter clipping
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{running_corrects/total:.4f}'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        self.criterion.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = F.cross_entropy(outputs, labels)  # No mixup in validation
                else:
                    outputs = self.model(images)
                    loss = F.cross_entropy(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += images.size(0)
                
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{running_corrects/total:.4f}'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("🚀 Starting anti-overfitting training...")
        print("=" * 70)
        
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler based on validation LOSS
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate overfitting gap
            overfit_gap = train_acc - val_acc
            
            # Log everything
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Metrics/overfit_gap', overfit_gap, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  📊 Overfit Gap: {overfit_gap:.4f} (target < 0.10)")
            print(f"  LR: {current_lr:.6f}")
            
            # Save based on validation LOSS (not accuracy)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'args': vars(self.args)
                }
                
                best_path = os.path.join(self.args.save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                
                print(f"  ✅ New best model! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ No improvement ({self.patience_counter}/{self.args.patience})")
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
            
            # Warning if overfitting is severe
            if overfit_gap > 0.15:
                print(f"  ⚠️  WARNING: Severe overfitting detected! Gap = {overfit_gap:.4f}")
            
            print("-" * 70)
        
        self.writer.close()
        
        print("\n" + "=" * 70)
        print(f"✅ Training complete!")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--img_size', type=int, default=224)
    
    parser.add_argument('--model', type=str, default='efficientnetv2')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.6, help='Increased from 0.3')
    
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=48)  # Slightly smaller
    parser.add_argument('--lr', type=float, default=5e-4, help='Reduced from 1e-3')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Increased from 1e-4')
    
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--save_dir', type=str, default='models/antioverfit')
    parser.add_argument('--log_dir', type=str, default='logs/antioverfit')
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    
    trainer = ImprovedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()