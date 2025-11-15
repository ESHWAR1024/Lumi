import torch
from torchvision import transforms
from PIL import Image
import io
from .model import get_resnet, SimpleCNN, get_efficientnetv2   # FIXED IMPORT

from .dataset import CLASS_NAMES, get_transforms
import numpy as np
import cv2


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

    # FIXED — efficientnet now uses get_efficientnetv2
    if arch == 'resnet':
        model = get_resnet(len(CLASS_NAMES), pretrained=False)

    elif arch == 'efficientnet':
        # EfficientNet wrapper → maps to v2 (your correct model)
        model = get_efficientnetv2(len(CLASS_NAMES), pretrained=False)

    elif arch == 'simple':
        model = SimpleCNN(len(CLASS_NAMES))

    else:
        raise ValueError(
            f"Unknown architecture: {arch}. Choose from 'resnet', 'efficientnet', or 'simple'"
        )
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    return model


def predict_from_pil(model, pil_img, device='cpu', img_size=48):
    """
    Predict emotion from PIL Image.
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


def detect_face(img_bytes):
    """
    Detect face in image and return cropped face.
    """

    # Convert bytes → numpy array → cv2 image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None

    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        return None

    # Largest face
    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    face_img = img[y:y+h, x:x+w]
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    pil_face = Image.fromarray(face_rgb)
    return pil_face


def predict_from_bytes(model, img_bytes, device='cpu', img_size=48, detect_face_first=True):
    """
    Predict emotion from raw image bytes.
    """

    if detect_face_first:
        face_pil = detect_face(img_bytes)

        if face_pil is None:
            pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            result = predict_from_pil(model, pil, device, img_size)
            result['face_detected'] = False
            return result
        else:
            result = predict_from_pil(model, face_pil, device, img_size)
            result['face_detected'] = True
            return result

    else:
        pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        result = predict_from_pil(model, pil, device, img_size)
        result['face_detected'] = None
        return result
