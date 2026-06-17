# 🔐 SecureVision – Fight Detection Model

## 📌 Overview
This repository contains the **Fight Detection Model** module of **SecureVision**, an AI-powered surveillance system designed to detect violent activities in real time using computer vision and deep learning.

This module focuses specifically on identifying **fight-related actions** from video input, making it useful for security monitoring in public places such as colleges, streets, malls, and offices.

It is a deep learning-based Fight Detection System developed using PyTorch and OpenCV for detecting violent activities in videos and live camera streams.

The project covers the complete machine learning pipeline, including dataset preparation, model training, model evaluation, cloud deployment on Microsoft Azure Virtual Machine, and client-side applications for both local and cloud-based inference.

---

It uses a Transformer-based video classification approach for fight detection.

The model is built upon the **TimeSformer (Time-Space Transformer)** architecture, which extends the Vision Transformer (ViT) paradigm to video understanding tasks.

Instead of processing individual frames independently, TimeSformer analyzes both:

- Spatial information (objects and actions within a frame)
- Temporal information (motion and activity across consecutive frames)

This enables the model to better understand violent interactions that unfold over time.

### Training Approach

The pre-trained TimeSformer model was fine-tuned for the binary classification task:

- Violence
- NonViolence

Fine-tuning was performed using custom violence detection datasets and additional negative mining samples to improve generalization and reduce false positives.

### Why TimeSformer?

Traditional CNN-based approaches often struggle to capture long-range temporal dependencies in videos.

TimeSformer leverages self-attention mechanisms to model temporal relationships between frames, making it well-suited for action recognition and fight detection tasks.

### Model Pipeline

```text
Input Video
      │
      ▼
Frame Sampling
      │
      ▼
TimeSformer Encoder
      │
      ▼
Temporal-Spatial Feature Extraction
      │
      ▼
Classification Head
      │
      ▼
Fight / Non-Fight Prediction
```

### Classification Task

| Class | Description |
|---------|------------|
| Violence | Physical violence or aggressive confrontation |
| NonViolence | Normal daily activities without violence |

Aur README ke top par short summary bhi add kar sakte ho:

# Fight Detection System

A Transformer-based video violence detection system built using PyTorch and TimeSformer. The model was fine-tuned on real-world violence datasets and deployed on Microsoft Azure Virtual Machine for cloud-based inference.

The system supports:

- Model training and experimentation using Google Colab
- Dataset documentation and management
- Local inference on a client machine
- Cloud deployment using Microsoft Azure VM
- API-based remote inference

---

## Repository Structure

```text
Fight-Detection-Model
│
├── 1. Model building (Google Colab)
│   ├── Fight_detection_model.ipynb
│   ├── Model_Testing.ipynb
│   ├── Negative_Mining.ipynb
│   └── model_demo.ipynb
│
├── 2. Dataset Used
│   └── Readme.md
│
├── 3. Client Demo (Local Inference)
│   ├── fight_model.py
│   ├── fight_predictor.py
│   ├── fight_inference.py
│   └── fight_webcam.py
│
├── 4. Azure Deployment (VM)
│   ├── app.py
│   ├── fight_model.py
│   └── fight_predictor.py
│
├── 5. Client Demo (Azure API)
│   └── demo.py
│
└── README.md
```

---

# Folder Details

## 1. Model Building (Google Colab)

This folder contains all notebooks used during model development and experimentation.

### Files

### Fight_detection_model.ipynb

Main training notebook responsible for:

- Dataset loading
- Data preprocessing
- Model training
- Model evaluation
- Model weight generation

### Model_Testing.ipynb

Used to evaluate model performance on test samples and unseen videos.

### Negative_Mining.ipynb

Contains experiments related to negative sample mining and dataset improvement.

### model_demo.ipynb

Demonstrates model inference and prediction results inside Google Colab.

---

## 2. Dataset Used

Contains documentation about all datasets used during training and model improvement.

The dataset documentation includes:

- Original training dataset
- Dataset sources
- Additional data collection
- Negative mining dataset
- Dataset references

Refer to:

```text
2. Dataset Used/Readme.md
```

for complete dataset details.

---

## 3. Client Demo (Local Inference)

This folder contains the local inference implementation.

The model runs directly on the user's machine without requiring any internet connection or cloud service.

### Files

#### fight_model.py

Contains the model architecture definition.

#### fight_predictor.py

Handles prediction logic and preprocessing.

#### fight_inference.py

Performs fight detection on videos.

#### fight_webcam.py

Performs real-time fight detection using a webcam.

### Workflow

```text
Input Video/Webcam
        │
        ▼
Model Loading
        │
        ▼
Prediction
        │
        ▼
Fight / Non-Fight Result
```

---

## 4. Azure Deployment (VM)

This folder contains the cloud deployment code used on Microsoft Azure Virtual Machine.

### Files

#### app.py

Flask API server responsible for serving prediction requests.

#### fight_model.py

Model architecture used by the server.

#### fight_predictor.py

Prediction pipeline used during API requests.

### Workflow

```text
Client Request
        │
        ▼
Azure API
        │
        ▼
Model Inference
        │
        ▼
Prediction Response
```

---

## 5. Client Demo (Azure API)

Contains a client application that communicates with the Azure-hosted model.

### File

#### demo.py

Sends requests to the Azure API and receives prediction results.

### Workflow

```text
Client Application
        │
        ▼
Azure API Server
        │
        ▼
Fight Detection Model
        │
        ▼
Prediction Response
```

---

# Model Weights

The trained model weights (`.pth`) are intentionally not included in this repository.

Reasons:

- Large file size
- Deployment constraints
- Model protection
- Repository size optimization

To run local inference or deploy the model independently, the trained model weights must be provided separately.

---

# Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Flask
- Google Colab
- Microsoft Azure Virtual Machine

---

# Deployment Architecture

```text
                ┌─────────────────┐
                │ Client Machine  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Azure API Server│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Fight Detection │
                │      Model      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Prediction      │
                └─────────────────┘
```

---

# Note

This repository is intended for educational, academic, and research purposes.

The repository demonstrates the complete development and deployment pipeline of a fight detection system, from dataset preparation and model training to cloud deployment and inference.

---

# Author

**Umar Farooque**

Bachelor of Engineering in Computer Engineering

M.H. Saboo Siddik College of Engineering, Mumbai
