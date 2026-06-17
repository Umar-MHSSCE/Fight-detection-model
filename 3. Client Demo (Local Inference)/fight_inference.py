import torch
from fight_model import load_fight_model
from fight_predictor import predict_fight_multiclip

VIDEO_PATH = "123.mp4"
MODEL_PATH = "models/securevision_fight_timesformer_hardneg.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading fight detection model...")
model = load_fight_model(MODEL_PATH, device)

print("Running inference on video...")
label, confidence = predict_fight_multiclip(VIDEO_PATH, model, device)

print(f"Prediction: {label} ({confidence:.2f})")
