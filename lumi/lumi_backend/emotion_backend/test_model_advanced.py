"""
Advanced model testing with:
- Test Time Augmentation (TTA)
- Confusion matrix visualization
- Per-class analysis
- Error analysis
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataset import FERImageFolder, get_advanced_transforms, TTADataset, CLASS_NAMES
from src.model import (get_efficientnetv2, get_convnext, get_swin_transformer, 
                       EnhancedEmotionCNN, get_ensemble_model)
from src.utils import compute_metrics, print_metrics, save_metrics_to_json
import argparse
from tqdm import tqdm
import os


def test_with_tta(model, dataset, device, n_tta=5):
    """
    Test with Test Time Augmentation
    Average predictions over multiple augmented versions
    """
    tta_dataset = TTADataset(dataset, n_tta=n_tta)
    loader = DataLoader(tta_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    
    # Store predictions for each original image
    predictions = {}
    labels = {}
    
    print(f"🔄 Running TTA with {n_tta} augmentations per image...")
    
    with torch.no_grad():
        for imgs, lbls, base_indices in tqdm(loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            
            # Accumulate predictions
            for i, base_idx in enumerate(base_indices):
                base_idx = base_idx.item()
                if base_idx not in predictions:
                    predictions[base_idx] = []
                    labels[base_idx] = lbls[i].item()
                predictions[base_idx].append(probs[i].cpu().numpy())
    
    # Average predictions
    y_true = []
    y_pred = []
    
    for idx in sorted(predictions.keys()):
        avg_pred = np.mean(predictions[idx], axis=0)
        pred_class = np.argmax(avg_pred)
        
        y_true.append(labels[idx])
        y_pred.append(pred_class)
    
    return y_true, y_pred


def test_standard(model, loader, device):
    """Standard testing without TTA"""
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    print("📊 Running standard evaluation...")
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
    
    return y_true, y_pred, y_probs


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to {save_path}")
    plt.close()


def analyze_errors(y_true, y_pred, y_probs, class_names, top_k=10):
    """Analyze most confident wrong predictions"""
    errors = []
    
    for i, (true_label, pred_label, probs) in enumerate(zip(y_true, y_pred, y_probs)):
        if true_label != pred_label:
            confidence = probs[pred_label]
            errors.append({
                'index': i,
                'true_label': class_names[true_label],
                'pred_label': class_names[pred_label],
                'confidence': confidence,
                'true_prob': probs[true_label]
            })
    
    # Sort by confidence (most confident mistakes)
    errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    print("\n" + "="*80)
    print("🔍 TOP CONFIDENT ERRORS")
    print("="*80)
    print(f"{'Index':<8} {'True':<12} {'Predicted':<12} {'Confidence':<12} {'True Prob':<12}")
    print("-"*80)
    
    for error in errors[:top_k]:
        print(f"{error['index']:<8} "
              f"{error['true_label']:<12} "
              f"{error['pred_label']:<12} "
              f"{error['confidence']:<12.4f} "
              f"{error['true_prob']:<12.4f}")
    
    print("="*80)
    
    return errors


def plot_per_class_metrics(metrics, class_names, save_path='per_class_metrics.png'):
    """Visualize per-class performance"""
    precisions = [metrics['per_class'][cls]['precision'] for cls in class_names]
    recalls = [metrics['per_class'][cls]['recall'] for cls in class_names]
    f1s = [metrics['per_class'][cls]['f1'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1s, width, label='F1 Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Per-class metrics saved to {save_path}")
    plt.close()


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print("="*80)
    print("🧪 ADVANCED MODEL TESTING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Architecture: {args.arch}")
    print(f"Use TTA: {args.use_tta}")
    print("="*80)
    
    # Load dataset
    test_ds = FERImageFolder(
        args.test_dir,
        transform=get_advanced_transforms('val', img_size=args.img_size)
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\n✅ Test dataset loaded: {len(test_ds)} images")
    
    # Load model
    print(f"\n🔄 Loading model...")
    
    if args.arch == 'efficientnetv2':
        model = get_efficientnetv2(len(CLASS_NAMES), pretrained=False)
    elif args.arch == 'convnext':
        model = get_convnext(len(CLASS_NAMES), pretrained=False)
    elif args.arch == 'swin':
        model = get_swin_transformer(len(CLASS_NAMES), pretrained=False)
    elif args.arch == 'enhanced':
        model = EnhancedEmotionCNN(len(CLASS_NAMES), pretrained=False)
    elif args.arch == 'ensemble':
        model = get_ensemble_model(len(CLASS_NAMES), pretrained=False)
    else:
        raise ValueError(f"Unknown architecture: {args.arch}")
    
    # Load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Test with or without TTA
    if args.use_tta:
        y_true, y_pred = test_with_tta(model, test_ds, device, n_tta=args.n_tta)
        y_probs = None
    else:
        y_true, y_pred, y_probs = test_standard(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    
    # Print metrics
    print_metrics(metrics, CLASS_NAMES)
    
    # Save metrics
    output_dir = os.path.dirname(args.model_path)
    metrics_file = os.path.join(output_dir, 'test_metrics.json')
    save_metrics_to_json(metrics, metrics_file)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, cm_path)
    
    # Plot per-class metrics
    pcm_path = os.path.join(output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics, CLASS_NAMES, pcm_path)
    
    # Analyze errors (if not using TTA)
    if y_probs is not None and args.analyze_errors:
        errors = analyze_errors(y_true, y_pred, y_probs, CLASS_NAMES, top_k=20)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    print(f"📊 Metrics saved to: {metrics_file}")
    print(f"📊 Confusion matrix: {cm_path}")
    print(f"📊 Per-class metrics: {pcm_path}")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Emotion Recognition Model')
    parser.add_argument('--test_dir', required=True, help='Test data directory')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--arch', required=True, 
                        choices=['efficientnetv2', 'convnext', 'swin', 'enhanced', 'ensemble'],
                        help='Model architecture')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--use_tta', action='store_true', help='Use Test Time Augmentation')
    parser.add_argument('--n_tta', type=int, default=5, help='Number of TTA iterations')
    parser.add_argument('--analyze_errors', action='store_true', default=True,
                        help='Analyze confident errors')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    
    args = parser.parse_args()
    main(args)