import torch
import os
from sklearn.metrics import accuracy_score, classification_report
import json

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    return checkpoint

def compute_accuracy(preds, targets):
    pred_labels = preds.argmax(dim=1).cpu().numpy()
    return accuracy_score(targets.cpu().numpy(), pred_labels)
