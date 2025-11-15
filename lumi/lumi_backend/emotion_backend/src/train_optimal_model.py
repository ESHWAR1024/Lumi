"""
Training script for the optimal emotion model
Perfectly tuned for 30-40k image datasets

This should give you:
- Train accuracy: 78-85%
- Val accuracy: 75-82%
- Gap: <5%
- No overfitting!
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
import random
import numpy as np

import sys
sys.path.append('src')
from dataset import FERImageFolder, get_transforms, CLASS_NAMES
from utils import save_checkpoint

# Import our optimal model
from optimal_emotion_model import get_optimal_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class OptimalTrainer:
    def __init__(self, args):
        set_seed(args.seed)
        
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print("🎯 OPTIMAL MODEL TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Dataset size category: {args.dataset_size}")
        print("=" * 70)
        
        self.setup_data()
        self.setup_model()
        self.setup_training()
        self.setup_logging()
    
    def setup_data(self):
        print("\n📦 Loading datasets...")
        
        train_transform = get_transforms('train', img_size=self.args.img_size)
        val_transform = get_transforms('val', img_size=self.args.img_size)
        
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
        
        # Verify dataset size matches selected model
        dataset_size = len(self.train_dataset)
        if self.args.dataset_size == 'small' and dataset_size > 20000:
            print(f"⚠️  Warning: Using 'small' model but have {dataset_size} images")
            print("   Consider using 'medium' model for better accuracy")
        elif self.args.dataset_size == 'large' and dataset_size < 40000:
            print(f"⚠️  Warning: Using 'large' model but only have {dataset_size} images")
            print("   Consider using 'medium' model to avoid overfitting")
        
        # Class weights for imbalanced data
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
        
        # DataLoaders
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
            batch_size=self.args.batch_size * 2,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
    
    def setup_model(self):
        print(f"\n🧠 Creating optimal model...")
        
        self.model = get_optimal_model(
            num_classes=len(CLASS_NAMES),
            dataset_size=self.args.dataset_size,
            dropout=self.args.dropout,
            pretrained=False
        )
        
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Total parameters: {total_params:,}")
        print(f"✅ Trainable: {trainable_params:,}")
        
        # Calculate model-to-data ratio
        ratio = len(self.train_dataset) / (total_params / 1000)
        print(f"📊 Data-to-params ratio: {ratio:.1f} images per 1k params")
        
        if ratio < 5:
            print("   ⚠️  Low ratio - might still overfit slightly")
        elif ratio < 10:
            print("   ✅ Good ratio - should generalize well")
        else:
            print("   ✅ Excellent ratio - strong generalization expected")
    
    def setup_training(self):
        print(f"\n⚙️ Setting up training...")
        
        # Loss function
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=0.1
            )
            print("✅ Using weighted CrossEntropy with label smoothing")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            print("✅ Using CrossEntropy with label smoothing")
        
        # Optimizer - single learning rate since no pretrained backbone
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999)
        )
        print(f"✅ AdamW optimizer: lr={self.args.lr}, wd={self.args.weight_decay}")
        
        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=15,
            T_mult=2,
            eta_min=self.args.lr * 0.01
        )
        print("✅ CosineAnnealingWarmRestarts scheduler")
        
        # Mixed precision
        self.scaler = GradScaler() if self.args.use_amp else None
        if self.args.use_amp:
            print("✅ Mixed precision enabled")
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def setup_logging(self):
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.args.log_dir)
        
        with open(os.path.join(self.args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        print(f"✅ Logging to: {self.args.log_dir}")
    
    def train_epoch(self, epoch):
        self.model.train()
        
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += images.size(0)
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{running_corrects/total:.4f}'
            })
        
        return running_loss / total, running_corrects / total
    
    def validate(self, epoch):
        self.model.eval()
        
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
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels).item()
                total += images.size(0)
                
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{running_corrects/total:.4f}'
                })
        
        return running_loss / total, running_corrects / total
    
    def train(self):
        print("\n" + "=" * 70)
        print("🚀 Starting training with optimal model...")
        print("=" * 70)
        
        for epoch in range(1, self.args.epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate metrics
            overfit_gap = train_acc - val_acc
            
            # Logging
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Metrics/overfit_gap', overfit_gap, epoch)
            self.writer.add_scalar('LR', current_lr, epoch)
            
            # Print results
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  📊 Gap: {overfit_gap:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Status indicator
            if overfit_gap < 0.03:
                print(f"  ✅ EXCELLENT - Minimal overfitting!")
            elif overfit_gap < 0.05:
                print(f"  ✅ GOOD - Low overfitting")
            elif overfit_gap < 0.08:
                print(f"  ⚠️  MODERATE - Acceptable overfitting")
            else:
                print(f"  ⚠️  HIGH - Still overfitting")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'overfit_gap': overfit_gap,
                    'args': vars(self.args)
                }
                
                best_path = os.path.join(self.args.save_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                
                print(f"  ✅ New best! Val Acc: {val_acc:.4f}, Gap: {overfit_gap:.4f}")
            else:
                self.patience_counter += 1
                print(f"  ⏳ No improvement ({self.patience_counter}/{self.args.patience})")
            
            # Save checkpoint periodically
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(self.args.save_dir, f'checkpoint_epoch{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state': self.model.state_dict(),
                    'val_acc': val_acc
                }, checkpoint_path)
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch}")
                break
            
            print("-" * 70)
        
        self.writer.close()
        
        print("\n" + "=" * 70)
        print(f"✅ Training complete!")
        print(f"   Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        print(f"   Model saved to: {self.args.save_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Train Optimal Emotion Model')
    
    # Data
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--img_size', type=int, default=224)
    
    # Model
    parser.add_argument('--dataset_size', type=str, default='medium',
                       choices=['small', 'medium', 'large'],
                       help='small (<20k), medium (20-40k), large (>40k)')
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='models/optimal_model')
    parser.add_argument('--log_dir', type=str, default='logs/optimal_model')
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    
    trainer = OptimalTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()