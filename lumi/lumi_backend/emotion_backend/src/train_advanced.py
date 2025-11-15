"""
Advanced training script with modern techniques
Place this file in: emotion_backend/src/train_advanced.py

Usage:
    python src/train_advanced.py --model efficientnetv2 --epochs 100 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np

# Import your modules
from dataset import FERImageFolder, get_transforms, CLASS_NAMES
from model_advanced import get_model
from utils import save_checkpoint


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Better than CrossEntropy for imbalanced datasets
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing for better generalization
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_pred = F.log_softmax(pred, dim=1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        return torch.mean(torch.sum(-true_dist * log_pred, dim=1))


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                    for base_lr in self.base_lrs]


class AdvancedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print(f"🚀 Advanced Emotion Recognition Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 70)
        
        # Setup
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.setup_logging()
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        print("\n📦 Loading datasets...")
        
        # Enhanced transforms
        train_transform = get_transforms('train', img_size=self.args.img_size)
        val_transform = get_transforms('val', img_size=self.args.img_size)
        
        # Datasets
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
        
        # Calculate class weights for imbalanced data
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
            print(f"✅ Class weights computed: {self.class_weights.tolist()}")
        else:
            self.class_weights = None
        
        # Dataloaders with optimizations
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if self.args.num_workers > 0 else False,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True if self.args.num_workers > 0 else False
        )
    
    def setup_model(self):
        """Setup model"""
        print(f"\n🧠 Creating model: {self.args.model}")
        
        self.model = get_model(
            model_name=self.args.model,
            num_classes=len(CLASS_NAMES),
            pretrained=self.args.pretrained
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """Setup loss, optimizer, scheduler"""
        print(f"\n⚙️ Setting up training components...")
        
        # Loss function
        if self.args.loss == 'focal':
            self.criterion = FocalLoss(alpha=1, gamma=2)
            print("✅ Using Focal Loss")
        elif self.args.loss == 'label_smoothing':
            self.criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
            print("✅ Using Label Smoothing CrossEntropy")
        else:
            if self.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                print("✅ Using Weighted CrossEntropy")
            else:
                self.criterion = nn.CrossEntropy Loss()
                print("✅ Using CrossEntropy")
        
        # Optimizer with different learning rates
        if self.args.pretrained:
            # Lower LR for backbone, higher for head
            backbone_params = []
            head_params = []
            
            for name, param in self.model.named_parameters():
                if 'classifier' in name or 'attention' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.args.lr * 0.1},
                {'params': head_params, 'lr': self.args.lr}
            ], weight_decay=self.args.weight_decay)
            
            print(f"✅ Differentiated LR: backbone={self.args.lr*0.1:.6f}, head={self.args.lr:.6f}")
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
            print(f"✅ Single LR: {self.args.lr:.6f}")
        
        # Learning rate scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.args.warmup_epochs,
            max_epochs=self.args.epochs,
            min_lr=1e-6
        )
        print(f"✅ Cosine LR scheduler with {self.args.warmup_epochs} warmup epochs")
        
        # Mixed precision training
        self.scaler = GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("✅ Mixed precision training enabled")
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
    
    def setup_logging(self):
        """Setup tensorboard and checkpoint directories"""
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        
        # Save training config
        config_path = os.path.join(self.args.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        print(f"✅ Logging to: {self.args.log_dir}")
        print(f"✅ Checkpoints to: {self.args.save_dir}")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Train]')
        
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            if self.args.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / total,
                'acc': running_corrects / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate model"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{self.args.epochs} [Val]')
        
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.args.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += images.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': running_loss / total,
                    'acc': running_corrects / total
                })
        
        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        
        # Calculate per-class metrics
        from sklearn.metrics import classification_report
        report = classification_report(
            all_labels,
            all_preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
        
        return epoch_loss, epoch_acc, report
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("🚀 Starting training...")
        print("=" * 70)
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, val_report = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Log per-class metrics
            for class_name in CLASS_NAMES:
                if class_name in val_report:
                    self.writer.add_scalar(
                        f'F1/{class_name}',
                        val_report[class_name]['f1-score'],
                        epoch
                    )
            
            # Print summary
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'args': vars(self.args)
                }
                
                best_path = os.path.join(self.args.save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                
                print(f"  ✅ New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ No improvement ({self.patience_counter}/{self.args.patience})")
            
            # Save checkpoint every N epochs
            if epoch % self.args.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.args.save_dir,
                    f'checkpoint_epoch{epoch}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'val_acc': val_acc
                }, checkpoint_path)
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                break
            
            print("-" * 70)
        
        # Training complete
        self.writer.close()
        
        print("\n" + "=" * 70)
        print(f"✅ Training complete!")
        print(f"   Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"   Best model: {os.path.join(self.args.save_dir, 'best_model.pt')}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Advanced Emotion Recognition Training')
    
    # Data
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--img_size', type=int, default=224)
    
    # Model
    parser.add_argument('--model', type=str, default='efficientnetv2',
                       choices=['efficientnetv2', 'efficientnetb3', 'resnetv2'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Loss
    parser.add_argument('--loss', type=str, default='focal',
                       choices=['ce', 'focal', 'label_smoothing'])
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='models/checkpoints_advanced')
    parser.add_argument('--log_dir', type=str, default='logs/advanced')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = AdvancedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()