import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import FERImageFolder, get_transforms, CLASS_NAMES
from model import SimpleCNN, get_resnet, get_resnet18, get_efficientnet_b0
from utils import save_checkpoint, compute_accuracy
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import os


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in tqdm(loader, desc='train'):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        running_acc += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='val'):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)
            running_acc += (out.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return running_loss / n, running_acc / n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("=" * 60)
    print("Using device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # FIXED: Use proper image size and transforms
    img_size = args.img_size if args.img_size > 0 else (224 if args.arch in ['resnet', 'resnet18', 'efficientnet'] else 48)
    print(f"Image size: {img_size}x{img_size}")
    
    train_ds = FERImageFolder(args.train_dir, transform=get_transforms('train', img_size=img_size))
    val_ds = FERImageFolder(args.val_dir, transform=get_transforms('val', img_size=img_size))
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        drop_last=True  # Drop incomplete batches for batch norm stability
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print("=" * 60)

    # FIXED: Model selection with proper configuration
    if args.arch == 'resnet':
        model = get_resnet(num_classes=len(CLASS_NAMES), pretrained=args.pretrained, freeze_backbone=False)
        print("Model: ResNet50 (all layers trainable)")
    elif args.arch == 'resnet18':
        model = get_resnet18(num_classes=len(CLASS_NAMES), pretrained=args.pretrained)
        print("Model: ResNet18 (lighter, better for 224x224)")
    elif args.arch == 'efficientnet':
        model = get_efficientnet_b0(num_classes=len(CLASS_NAMES), pretrained=args.pretrained)
        print("Model: EfficientNet-B0 (recommended for emotion recognition)")
    else:
        model = SimpleCNN(num_classes=len(CLASS_NAMES))
        print("Model: SimpleCNN (from scratch)")
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    # FIXED: Better loss function for class imbalance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing helps generalization
    
    # FIXED: Differentiated learning rates
    if args.arch in ['resnet', 'resnet18', 'efficientnet'] and args.pretrained:
        # Lower LR for pretrained backbone, higher for new classifier
        backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n and 'classifier' not in n]
        head_params = [p for n, p in model.named_parameters() if 'fc' in n or 'classifier' in n]
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': args.lr * 0.1},  # 10x lower for backbone
            {'params': head_params, 'lr': args.lr}
        ], weight_decay=1e-4)
        print(f"Using differentiated LR: backbone={args.lr*0.1:.6f}, head={args.lr:.6f}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"Using single LR: {args.lr:.6f}")
    
    # IMPROVED: Better learning rate schedule
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Monitor validation accuracy
        factor=0.5, 
        patience=5,  # Reduce LR if no improvement for 5 epochs
        
    )

    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc
            }, path)
            print(f"✅ Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs (no improvement for {early_stop_patience} epochs)")
            break
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, path)

    writer.close()
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--save_dir', default='models/checkpoints')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)  # Reduced from 64 for 224x224
    parser.add_argument('--lr', type=float, default=3e-4)  # Lower initial LR
    parser.add_argument('--img_size', type=int, default=224)  # Changed default to 224
    parser.add_argument('--arch', choices=['simple', 'resnet', 'resnet18', 'efficientnet'], default='resnet18')
    parser.add_argument('--pretrained', action='store_true', default=True)  # Default to pretrained
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    main(args)