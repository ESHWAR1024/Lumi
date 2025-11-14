from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .inference import load_model, predict_from_bytes
import os
import torch
import base64

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/checkpoints/checkpoint_epoch50.pt')
DEVICE = 'cuda' if torch.cuda.is_available() and os.environ.get('USE_CUDA','1') == '1' else 'cpu'

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add your Next.js ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model = load_model(MODEL_PATH, device=DEVICE, arch='efficientnet')
    print(f"✅ Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None


@app.get('/')
def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "device": DEVICE
    }


@app.get('/health')
def health():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }


@app.post('/predict-image')
async def predict_image(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded image.
    Supports: JPG, JPEG, PNG
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        content = await file.read()
        result = predict_from_bytes(model, content, device=DEVICE, img_size=224)
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post('/predict-frame')
async def predict_frame(file: UploadFile = File(...)):
    """
    Real-time prediction endpoint for webcam frames.
    Optimized for speed.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        content = await file.read()
        result = predict_from_bytes(model, content, device=DEVICE, img_size=224)
        
        return {
            "success": True,
            "emotion": result['label'],
            "confidence": result['prob'],
            "all_probabilities": result['all_probs']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post('/predict-batch')
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict emotions from multiple images.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            content = await file.read()
            result = predict_from_bytes(model, content, device=DEVICE, img_size=224)
            results.append({
                "filename": file.filename,
                "prediction": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {"results": results, "total": len(results)}


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)