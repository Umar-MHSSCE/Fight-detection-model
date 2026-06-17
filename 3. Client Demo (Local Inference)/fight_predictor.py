import torch
import numpy as np
from torchvision import transforms
from decord import VideoReader, cpu

# Same preprocessing as training - Double check if normalization was used in training!
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_clip_frames(vr, indices):
    frames = vr.get_batch(indices).asnumpy()
    frames = torch.stack([transform(f) for f in frames])  # (T, C, H, W)
    return frames

def predict_fight_multiclip(video_path, model, device, 
                            clips=5, conf_thresh=0.50, min_hits=2):
    """
    Lowered conf_thresh to 0.50 and min_hits to 2 to be less restrictive.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    violence_probs = []

    # Calculate indices using Sparse Sampling
    # We want to cover more than just a fraction of a second
    for i in range(clips):
        # Sample 8 frames spread across the video segment
        if total_frames > 30:
            start = int(i * (total_frames - 30) / clips)
            end = start + 30
            indices = np.linspace(start, min(end, total_frames - 1), 8).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, 8).astype(int)

        clip = load_clip_frames(vr, indices)
        clip = clip.unsqueeze(0).to(device) # (1, T, C, H, W)

        with torch.no_grad():
            logits = model(clip)
            probs = torch.softmax(logits, dim=1)
            # Probability of class 1 (Violence)
            violence_probs.append(probs[0, 1].item())

    # Decision Logic
    hits = sum(1 for p in violence_probs if p > conf_thresh)
    avg_conf = sum(violence_probs) / len(violence_probs)
    
    # DEBUG: See what the model is thinking for every clip
    print(f"Raw Probabilities: {[round(p, 3) for p in violence_probs]}")

    if hits >= min_hits:
        return "Violence", max(violence_probs)
    else:
        return "NonViolence", avg_conf