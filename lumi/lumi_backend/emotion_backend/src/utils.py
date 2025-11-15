import torch
import torch.nn as nn
import os
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             classification_report, confusion_matrix)
import json
import numpy as np


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"💾 Checkpoint saved: {filename}")


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute comprehensive metrics
    
    Returns:
        dict with accuracy, precision, recall, f1, per-class metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted metrics (accounts for class imbalance)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = \
        precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': {
            class_names[i]: {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int(support[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def print_metrics(metrics, class_names):
    """Pretty print metrics"""
    print("\n" + "="*80)
    print("📊 EVALUATION METRICS")
    print("="*80)
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1 Score:  {metrics['f1']:.4f}")
    
    print(f"\n📈 Per-Class Performance:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print("-" * 60)
    
    for class_name in class_names:
        metrics_class = metrics['per_class'][class_name]
        print(f"{class_name:<12} "
              f"{metrics_class['precision']:<12.4f} "
              f"{metrics_class['recall']:<12.4f} "
              f"{metrics_class['f1']:<12.4f} "
              f"{metrics_class['support']:<12}")
    print("="*80)


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    Stops training when validation metric stops improving
    """
    def __init__(self, patience=10, verbose=True, delta=0, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            verbose: Print messages
            delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy, 'min' for loss
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch=None):
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"⚠️  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and score > self.best_score:
                improvement = score - self.best_score
                print(f"✅ Metric improved by {improvement:.4f}")
            self.best_score = score
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses training on hard examples
    
    Reference: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Weighting factor in [0,1] to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Prevents overconfident predictions
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def compute_class_weights(dataset, num_classes):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
    
    Returns:
        torch.Tensor of class weights
    """
    class_counts = torch.zeros(num_classes)
    
    for _, label in dataset:
        class_counts[label] += 1
    
    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return class_weights


def mixup_data(x, y, alpha=1.0, device='cuda'):
    """
    Apply mixup augmentation
    
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_metrics_to_json(metrics, filepath):
    """Save metrics dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to {filepath}")


def load_metrics_from_json(filepath):
    """Load metrics from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_layers(model, freeze_bn=True):
    """Freeze all layers except the final classifier"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier (last layer)
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Optionally freeze batch norm layers
    if freeze_bn:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


def unfreeze_layers(model, unfreeze_after_layer=None):
    """Progressively unfreeze layers"""
    if unfreeze_after_layer is None:
        # Unfreeze all
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze layers after specific layer
        unfreeze = False
        for name, param in model.named_parameters():
            if unfreeze_after_layer in name:
                unfreeze = True
            if unfreeze:
                param.requires_grad = True


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def setup_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("⚠️  GPU not available, using CPU")
    return device