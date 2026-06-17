from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import base64
import cv2
from fight_model import load_fight_model
from fight_predictor import transform

app = FastAPI()

MODEL_PATH = "models/securevision_fight_timesformer_hardneg.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once at startup
model = load_fight_model(MODEL_PATH, device)

class FrameRequest(BaseModel):
    frames: list # List of base64 strings

@app.post("/predict-frames")
def predict_frames(data: FrameRequest):
    processed_frames = []
    
    for f in data.frames:
        img_bytes = base64.b64decode(f)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            processed_frames.append(transform(img))

    if len(processed_frames) < 8:
        return {"error": "Need 8 frames", "violence_prob": 0.0}

    # Stack into (1, T, C, H, W)
    frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)
    
    violence_prob = probs[0, 1].item()
    
    return {
        "violence_prob": float(violence_prob),
        "status": "Violence" if violence_prob > 0.5 else "NonViolence"
    }