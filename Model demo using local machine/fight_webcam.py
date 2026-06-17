import torch
import cv2
import numpy as np
from fight_model import load_fight_model
from fight_predictor import transform

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/securevision_fight_timesformer_hardneg.pth"

CLIP_SIZE = 8          
TOTAL_CLIPS = 3        # Reduced for faster feedback
FRAME_SKIP = 5         # Increased to capture more 'action' over time
CONF_THRESH = 0.50     
MIN_HITS = 1           # If even one clip looks like violence, we alert

# -----------------------------
# LOAD MODEL
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
model = load_fight_model(MODEL_PATH, device)

cap = cv2.VideoCapture(0)
frame_buffer = []
violence_probs = []
frame_count = 0
label = "Analyzing..."
confidence = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame_count += 1
    # Only keep every Nth frame to create a 'motion' clip
    if frame_count % FRAME_SKIP == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)

    if len(frame_buffer) == CLIP_SIZE:
        # Preprocess the buffer
        input_tensor = torch.stack([transform(f) for f in frame_buffer])
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        
        v_prob = probs[0, 1].item()
        violence_probs.append(v_prob)
        frame_buffer = [] # Clear for next clip

        # Real-time update
        if len(violence_probs) >= TOTAL_CLIPS:
            hits = sum(p > CONF_THRESH for p in violence_probs)
            label = "Violence" if hits >= MIN_HITS else "NonViolence"
            confidence = max(violence_probs)
            violence_probs.pop(0) # Sliding window voting

    # UI Overlay
    color = (0, 0, 255) if label == "Violence" else (0, 255, 0)
    cv2.putText(frame, f"STATUS: {label} ({confidence:.2f})", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("SecureVision AI - Fight Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()