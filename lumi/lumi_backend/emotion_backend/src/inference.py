import torch
from torchvision import transforms
from PIL import Image
import io
from model import get_resnet, SimpleCNN
from dataset import CLASS_NAMES, get_transforms
import numpy as np

def load_model(path, device='cpu', arch='resnet'):
    if arch == 'resnet':
        model = get_resnet(len(CLASS_NAMES), pretrained=False)
    else:
        model = SimpleCNN(len(CLASS_NAMES))
    model.load_state_dict(torch.load(path, map_location=device)['model_state'])
    model.to(device)
    model.eval()
    return model

def predict_from_pil(model, pil_img, device='cpu', img_size=48):
    tf = get_transforms('val', img_size=img_size)
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = probs.argmax()
    return {'label': CLASS_NAMES[pred], 'prob': float(probs[pred]), 'all_probs': {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}}

def predict_from_bytes(model, img_bytes, device='cpu', img_size=48):
    pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return predict_from_pil(model, pil, device, img_size)
