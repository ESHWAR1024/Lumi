from fastapi import FastAPI, File, UploadFile
import uvicorn
from inference import load_model, predict_from_bytes
import os

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/checkpoints/best_model.pt')
DEVICE = 'cuda' if os.environ.get('USE_CUDA','0') == '1' else 'cpu'

app = FastAPI()
model = load_model(MODEL_PATH, device=DEVICE, arch='resnet')

@app.get('/')
def root():
    return {"status": "running"}

@app.post('/predict-image')
async def predict_image(file: UploadFile = File(...)):
    content = await file.read()
    res = predict_from_bytes(model, content, device=DEVICE, img_size=48)
    return res

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
