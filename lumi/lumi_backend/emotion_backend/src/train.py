import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from dataset import FERImageFolder, get_transforms, CLASS_NAMES
from model import SimpleCNN, get_resnet
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
    print("Using device:", device)

    train_ds = FERImageFolder(args.train_dir, transform=get_transforms('train', img_size=args.img_size))
    val_ds = FERImageFolder(args.val_dir, transform=get_transforms('val', img_size=args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    if args.arch == 'resnet':
        model = get_resnet(num_classes=len(CLASS_NAMES), pretrained=args.pretrained)
    else:
        model = SimpleCNN(num_classes=len(CLASS_NAMES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(log_dir=args.log_dir)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | Val loss: {val_loss:.4f} acc: {val_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = os.path.join(args.save_dir, f'best_model_epoch{epoch}_acc{val_acc:.4f}.pt')
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, path)
            print("Saved best model to", path)

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--save_dir', default='models/checkpoints')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=48)
    parser.add_argument('--arch', choices=['simple','resnet'], default='resnet')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()
    main(args)
