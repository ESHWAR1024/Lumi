# Emotion Backend - Emotion Recognition CNN

A deep learning project for emotion recognition from facial images using PyTorch and FastAPI.

## Features

- **Model Architecture**: Pre-trained ResNet50 backbone with custom classifier head
- **Data Pipeline**: Custom PyTorch Dataset with augmentation transforms
- **Training Pipeline**: Full training and validation loop with checkpointing
- **API Server**: FastAPI-based REST API for inference
- **Inference**: Single image and batch prediction support

## Directory Structure

```
emotion_backend/
├─ data/
│  ├─ train/                # Training images organized by emotion class
│  ├─ val/                  # Validation images
│  └─ test/                 # Test split (optional)
├─ models/
│  └─ checkpoints/          # Saved .pt model files
├─ src/
│  ├─ __init__.py           # Package initialization
│  ├─ dataset.py            # Custom PyTorch Dataset + transforms
│  ├─ model.py              # CNN model architectures
│  ├─ train.py              # Training + validation loop
│  ├─ utils.py              # Helper functions (metrics, save/load)
│  ├─ inference.py          # Inference helper for single image
│  └─ app.py                # FastAPI server
├─ logs/                    # Training logs and metrics
├─ requirements.txt         # Python dependencies
├─ README.md               # This file
└─ run_train.sh            # Training script
```

## Installation

1. **Clone the repository and navigate to the project:**
```bash
cd emotion_backend
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Data Preparation

Organize your dataset in the following structure:

```
data/
├─ train/
│  ├─ angry/
│  ├─ disgust/
│  ├─ fear/
│  ├─ happy/
│  ├─ neutral/
│  ├─ sad/
│  └─ surprise/
├─ val/
│  └─ [same structure as train]
└─ test/
   └─ [same structure as train]
```

## Training

### Using Python Script

```python
from src.train import Trainer
from src.utils import get_device, setup_seed

# Set seed for reproducibility
setup_seed(42)

# Initialize trainer
device = get_device()
trainer = Trainer(model_name='resnet50', num_classes=7, device=str(device))

# Train model
history = trainer.train(
    train_data_dir='./data/train',
    val_data_dir='./data/val',
    epochs=50,
    batch_size=32,
    lr=0.001,
    checkpoint_dir='./models/checkpoints',
    log_dir='./logs'
)
```

### Using Bash Script

```bash
bash run_train.sh
```

## Inference

### Single Image Prediction

```python
from src.inference import EmotionPredictor

predictor = EmotionPredictor(
    checkpoint_path='./models/checkpoints/best_model.pt',
    model_name='resnet50',
    num_classes=7,
    device='cuda'
)

result = predictor.predict('path/to/image.jpg')
print(result)
# Output: {'emotion': 'happy', 'confidence': 0.95, 'all_probabilities': {...}}
```

### Batch Prediction

```python
results = predictor.predict_batch(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

## API Server

### Start the API Server

```bash
python -m src.app
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### 1. Health Check
```
GET /
GET /health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device_info": {
    "cuda_available": true,
    "device": "GPU",
    "gpu_name": "NVIDIA RTX 3090"
  }
}
```

#### 2. Single Image Prediction
```
POST /predict
```

Request: Multipart form with image file
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```

Response:
```json
{
  "success": true,
  "filename": "image.jpg",
  "prediction": {
    "emotion": "happy",
    "confidence": 0.95,
    "class_id": 3,
    "all_probabilities": {
      "Angry": 0.01,
      "Disgust": 0.01,
      "Fear": 0.01,
      "Happy": 0.95,
      "Neutral": 0.01,
      "Sad": 0.01,
      "Surprise": 0.01
    }
  }
}
```

#### 3. Batch Prediction
```
POST /predict/batch
```

Request: Multipart form with multiple image files
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## Supported Emotion Classes

1. **Angry** - Expression of anger or hostility
2. **Disgust** - Expression of disapproval or revulsion
3. **Fear** - Expression of anxiety or concern
4. **Happy** - Expression of joy or happiness
5. **Neutral** - Neutral or expressionless face
6. **Sad** - Expression of sadness or sorrow
7. **Surprise** - Expression of surprise or astonishment

## Model Architecture

### EmotionCNN (Default)
- **Backbone**: Pre-trained ResNet50
- **Classifier**: Custom linear layer (2048 → num_classes)
- **Features**: Transfer learning, pre-trained weights
- **Best For**: High accuracy with limited data

### SimpleCNN
- **Architecture**: Custom lightweight CNN
- **Features**: Fast inference, lower memory requirements
- **Best For**: Real-time applications, edge devices

## Configuration

You can configure the model by setting environment variables:

```bash
export MODEL_CHECKPOINT="./models/checkpoints/best_model.pt"
export PORT=8000
python -m src.app
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU support)
- 4GB+ RAM (8GB+ recommended for GPU)

## Performance Metrics

The model tracks the following metrics during training:

- **Accuracy**: Overall classification accuracy
- **Loss**: Cross-entropy loss
- **Precision**: Per-class precision (weighted average)
- **Recall**: Per-class recall (weighted average)
- **F1 Score**: Harmonic mean of precision and recall

Training history is saved in `logs/training_history.json`

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch_size=16`
- Use CPU: `device='cpu'`
- Clear GPU cache: `torch.cuda.empty_cache()`

### Model Not Found
- Ensure checkpoint exists at specified path
- Check file permissions
- Verify model name matches architecture

### API Connection Issues
- Ensure port 8000 is not in use
- Check firewall settings
- Use `--host 0.0.0.0` to allow external connections

## License

MIT License - Feel free to use for academic and commercial purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## References

- ResNet: He, K., et al. (2015). "Deep Residual Learning for Image Recognition"
- PyTorch: https://pytorch.org/
- FastAPI: https://fastapi.tiangolo.com/
