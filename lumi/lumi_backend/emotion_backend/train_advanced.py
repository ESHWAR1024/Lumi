import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from dataset import FERImageFolder, get_advanced_transforms, MixUp, CutMix, CLASS_NAMES
from model import (EnhancedEmotionCNN, get_efficientnetv2, get_convnext, 
                   get_swin_transformer, get_ensemble_model)
from utils import save_checkpoint, EarlyStopping, FocalLoss, compute_metrics
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import os
import random
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, 
                     use_mixup=True, use_cutmix=True, epoch=0):
    """
    Training with advanced techniques:
    - Mixed precision training
    - MixUp/CutMix augmentation
    - Gradient accumulation
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    
    mixup = MixUp(alpha=0.2) if use_mixup else None
    cutmix = CutMix(alpha=1.0) if use_cutmix else None
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Apply MixUp or CutMix randomly
        use_mix = random.random() < 0.5
        if use_mix and mixup is not None and random.random() < 0.5:
            imgs, labels_a, labels_b, lam = mixup(imgs, labels)
            mixed = True
        elif use_mix and cutmix is not None:
            imgs, labels_a, labels_b, lam = cutmix(imgs, labels)
            mixed = True
        else:
            mixed = False
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(imgs)
            
            if mixed:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Statistics
        running_loss += loss.item() * imgs.size(0)
        total += labels.size(0)
        
        if not mixed:
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        else:
            # Approximate accuracy for mixed samples
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels_a.data) * 0.5
            running_corrects += torch.sum(preds == labels_b.data) * 0.5
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{running_corrects.double()/total:.4f}'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = running_corrects.double() / total
    
    return epoch_loss, epoch_acc.item()


def validate(model, loader, criterion, device, return_predictions=False):
    """Validation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='[Val]'):
            imgs, labels = imgs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, CLASS_NAMES)
    
    if return_predictions:
        return epoch_loss, metrics, all_preds, all_labels
    return epoch_loss, metrics


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("=" * 80)
    print(f"🚀 ADVANCED EMOTION RECOGNITION TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Architecture: {args.arch}")
    print("=" * 80)
    
    # Dataset with advanced augmentations
    train_ds = FERImageFolder(
        args.train_dir, 
        transform=get_advanced_transforms('train', img_size=args.img_size),
        balance_classes=args.balance_classes
    )
    val_ds = FERImageFolder(
        args.val_dir, 
        transform=get_advanced_transforms('val', img_size=args.img_size)
    )
    
    # Data loaders with optimizations
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    print(f"\n📊 Dataset Info:")
    print(f"   Training samples: {len(train_ds)}")
    print(f"   Validation samples: {len(val_ds)}")
    print(f"   Training batches: {len(train_loader)}")
    print("=" * 80)
    
    # Model selection
    print(f"\n🧠 Initializing {args.arch} model...")
    
    if args.arch == 'enhanced':
        model = EnhancedEmotionCNN(len(CLASS_NAMES), pretrained=args.pretrained, dropout=args.dropout)
    elif args.arch == 'efficientnetv2':
        model = get_efficientnetv2(len(CLASS_NAMES), pretrained=args.pretrained, model_size='s')
    elif args.arch == 'convnext':
        model = get_convnext(len(CLASS_NAMES), pretrained=args.pretrained, model_size='tiny')
    elif args.arch == 'swin':
        model = get_swin_transformer(len(CLASS_NAMES), pretrained=args.pretrained, model_size='tiny')
    elif args.arch == 'ensemble':
        model = get_ensemble_model(len(CLASS_NAMES), pretrained=args.pretrained)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model loaded!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print("=" * 80)
    
    # Loss function - Focal Loss for class imbalance
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print("📊 Using Focal Loss (handles class imbalance)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"📊 Using CrossEntropyLoss (label_smoothing={args.label_smoothing})")
    
    # Optimizer - AdamW with weight decay
    if args.use_sam:
        # Sharpness-Aware Minimization (SAM) for better generalization
        from sam import SAM
        base_optimizer = optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, weight_decay=args.weight_decay)
        print("🔧 Using SAM optimizer (better generalization)")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print("🔧 Using AdamW optimizer")
    
    # Learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.t0, T_mult=2, eta_min=args.lr * 0.01
        )
        print(f"📈 Using CosineAnnealingWarmRestarts (T_0={args.t0})")
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_loader)
        )
        print("📈 Using OneCycleLR")
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        print("📈 Using ReduceLROnPlateau")
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Stochastic Weight Averaging (SWA) for better generalization
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.lr * 0.1)
        swa_start = int(args.epochs * 0.75)
        print(f"🔄 Using SWA (starts at epoch {swa_start})")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    print("\n" + "=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_mixup=args.use_mixup, use_cutmix=args.use_cutmix, epoch=epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        
        # Update SWA
        if args.use_swa and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif args.scheduler == 'onecycle':
            scheduler.step()
        elif args.scheduler != 'plateau':
            scheduler.step()
        else:
            scheduler.step(val_acc)
        
        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📊 Results:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"   Val Precision: {val_metrics['precision']:.4f}")
        print(f"   Val Recall: {val_metrics['recall']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}")
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Metrics/precision', val_metrics['precision'], epoch)
        writer.add_scalar('Metrics/recall', val_metrics['recall'], epoch)
        writer.add_scalar('Metrics/f1', val_metrics['f1'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'metrics': val_metrics,
                'args': vars(args)
            }, save_path)
            print(f"✅ Saved best model (val_acc: {val_acc:.4f})")
        
        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break
        
        # Save checkpoint every N epochs
        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, save_path)
    
    # Final SWA update
    if args.use_swa:
        print("\n🔄 Updating SWA batch normalization statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
        
        # Evaluate SWA model
        print("📊 Evaluating SWA model...")
        val_loss, val_metrics = validate(swa_model, val_loader, criterion, device)
        swa_acc = val_metrics['accuracy']
        
        print(f"SWA Model - Val Acc: {swa_acc:.4f}")
        
        if swa_acc > best_val_acc:
            save_path = os.path.join(args.save_dir, 'best_model_swa.pt')
            save_checkpoint({
                'model_state': swa_model.module.state_dict(),
                'val_acc': swa_acc,
                'metrics': val_metrics
            }, save_path)
            print(f"✅ SWA model is better! Saved to {save_path}")
    
    writer.close()
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"📁 Models saved to: {args.save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Emotion Recognition Training')
    
    # Data
    parser.add_argument('--train_dir', required=True, help='Training data directory')
    parser.add_argument('--val_dir', required=True, help='Validation data directory')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--balance_classes', action='store_true', help='Balance classes by oversampling')
    
    # Model
    parser.add_argument('--arch', default='efficientnetv2', 
                        choices=['enhanced', 'efficientnetv2', 'convnext', 'swin', 'ensemble'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Loss
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    
    # Optimizer
    parser.add_argument('--use_sam', action='store_true', help='Use SAM optimizer')
    
    # Scheduler
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'onecycle', 'plateau'])
    parser.add_argument('--t0', type=int, default=10, help='CosineAnnealing T_0')
    
    # Augmentation
    parser.add_argument('--use_mixup', action='store_true', default=True, help='Use MixUp')
    parser.add_argument('--use_cutmix', action='store_true', default=True, help='Use CutMix')
    
    # Advanced techniques
    parser.add_argument('--use_swa', action='store_true', help='Use Stochastic Weight Averaging')
    
    # Save/Load
    parser.add_argument('--save_dir', default='models/checkpoints_advanced', help='Save directory')
    parser.add_argument('--log_dir', default='logs/advanced', help='TensorBoard log directory')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint frequency')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    # Other
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)