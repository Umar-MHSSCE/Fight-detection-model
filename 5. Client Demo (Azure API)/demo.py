import cv2
import requests
import base64
import threading
import time

API_URL = "http://20.2.235.183:8000/predict-frames"

cap = cv2.VideoCapture(0)
frame_buffer = []
latest_prediction = "Analyzing..."
latest_prob = 0.0
is_processing = False

def call_api(frames_to_send):
    global latest_prediction, latest_prob, is_processing
    try:
        # 1. Increased timeout to 10 seconds because TimeSformer is heavy
        res = requests.post(API_URL, json={"frames": frames_to_send}, timeout=10)
        data = res.json()
        latest_prob = data["violence_prob"]
        latest_prediction = data["status"]
    except Exception as e:
        print(f"API Error: {e}")
    finally:
        # 2. Add a tiny delay so we don't spam the server immediately again
        time.sleep(0.5) 
        is_processing = False

print("Starting SecureVision Stream...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Show the frame IMMEDIATELY (No lag)
    display_frame = frame.copy()
    color = (0, 0, 255) if latest_prob > 0.5 else (0, 255, 0)
    cv2.putText(display_frame, f"STATUS: {latest_prediction} ({latest_prob:.2f})", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("SecureVision - Azure Cloud Inference", display_frame)

    # 2. Collect frames for the buffer (Resize to 224 to save bandwidth!)
    small_frame = cv2.resize(frame, (224, 224))
    _, buffer = cv2.imencode('.jpg', small_frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    
    frame_buffer.append(frame_b64)
    if len(frame_buffer) > 8:
        frame_buffer.pop(0)

    # 3. Send to Azure only if we aren't already waiting for a response
    # This prevents the "clogging" that makes performance bad
    if len(frame_buffer) == 8 and not is_processing:
        is_processing = True
        # Send in a background thread so the webcam doesn't freeze
        threading.Thread(target=call_api, args=(frame_buffer.copy(),)).start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()