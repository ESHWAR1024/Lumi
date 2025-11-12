import torch
from torchvision import transforms
from PIL import Image
import io
from .model import get_resnet, SimpleCNN, get_efficientnet
from .dataset import CLASS_NAMES, get_transforms
import numpy as np

def load_model(path, device='cpu', arch='resnet'):
    """
    Load trained model from checkpoint.
    
    Args:
        path: Path to model checkpoint (.pt file)
        device: Device to load model on ('cpu' or 'cuda')
        arch: Architecture type ('resnet', 'efficientnet', or 'simple')
    
    Returns:
        Loaded model in evaluation mode
    """
    if arch == 'resnet':
        model = get_resnet(len(CLASS_NAMES), pretrained=False)
    elif arch == 'efficientnet':
        model = get_efficientnet(len(CLASS_NAMES), pretrained=False)
    elif arch == 'simple':
        model = SimpleCNN(len(CLASS_NAMES))
    else:
        raise ValueError(f"Unknown architecture: {arch}. Choose from 'resnet', 'efficientnet', or 'simple'")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def predict_from_pil(model, pil_img, device='cpu', img_size=48):
    """
    Predict emotion from PIL Image.
    
    Args:
        model: Loaded PyTorch model
        pil_img: PIL Image object
        device: Device for inference
        img_size: Image size for preprocessing
    
    Returns:
        Dictionary with 'label', 'prob', and 'all_probs'
    """
    tf = get_transforms('val', img_size=img_size)
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    return {
        'label': CLASS_NAMES[pred], 
        'prob': float(probs[pred]), 
        'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }

def predict_from_bytes(model, img_bytes, device='cpu', img_size=48):
    """
    Predict emotion from image bytes.
    
    Args:
        model: Loaded PyTorch model
        img_bytes: Image data as bytes
        device: Device for inference
        img_size: Image size for preprocessing
    
    Returns:
        Dictionary with 'label', 'prob', and 'all_probs'
    """
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return predict_from_pil(model, pil, device, img_size)