"""
Evaluate trained FER2013 model on test dataset.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.dataset import FERImageFolder, get_transforms
from src.model import get_resnet, SimpleCNN
from src.utils import compute_accuracy

# ----------------------------
# 🔧 Configuration
# ----------------------------
TEST_DIR = "data/test"
MODEL_PATH = "models/checkpoints/best_model_epoch1_acc0.42.pt"  # 👈 Replace with your best .pt model
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# 🔍 Dataset setup
# ----------------------------
print("🔹 Loading test dataset from:", TEST_DIR)
test_dataset = FERImageFolder(TEST_DIR, transform=get_transforms("val", img_size=48))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
CLASS_NAMES = test_dataset.classes
print("Classes:", CLASS_NAMES)

# ----------------------------
# 🧠 Load trained model
# ----------------------------
print("🔹 Loading model:", MODEL_PATH)
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load checkpoint at {MODEL_PATH}: {e}")

if "model_state" in checkpoint:
    model_state = checkpoint["model_state"]
else:
    model_state = checkpoint

# choose architecture (same used in training)
model = get_resnet(num_classes=len(CLASS_NAMES), pretrained=False)
model.load_state_dict(model_state)
model.to(DEVICE)
model.eval()

# ----------------------------
# 📊 Evaluation
# ----------------------------
y_true, y_pred = [], []

print("🔹 Evaluating...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("✅ Evaluation complete.")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

acc = np.trace(cm) / np.sum(cm)
print(f"\n✅ Overall Accuracy: {acc * 100:.2f}%")
