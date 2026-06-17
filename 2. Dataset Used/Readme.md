# Fight detection model - Dataset Documentation

This document describes the datasets used for training, testing, and improving the fight detection model used in fight detection model of SecureVision AI.

---

# 1. Initial Training Dataset

The fight detection model was initially trained using the Real Life Violence Situations Dataset.

Dataset:
- Real Life Violence Situations Dataset

Source:
- https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset

This dataset contains real-world violence and non-violence video clips and served as the primary dataset for model training.

---

# 2. Testing Dataset

A custom testing dataset was created by combining videos from multiple publicly available datasets to evaluate the model on unseen data.

## Non-Violence Videos

| Source | Videos Used |
|----------|----------|
| Movies Fight Detection Dataset | 100 |
| AirtLab Dataset (CAM2) | 60 |

Total Non-Violence Videos: **160**

### Sources

- Movies Fight Detection Dataset  
  https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset

- AirtLab Dataset for Automatic Violence Detection in Videos  
  https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos

---

## Violence Videos

| Source | Videos Used |
|----------|----------|
| CCTV Fights Dataset | 70 |
| Movies Fight Detection Dataset | 80 |

Total Violence Videos: **150**

### Sources

- CCTV Fights Dataset  
  https://www.kaggle.com/datasets/shreyj1729/cctv-fights-datase

- Movies Fight Detection Dataset  
  https://www.kaggle.com/datasets/naveenk903/movies-fight-detection-dataset

---

## Testing Dataset Summary

| Class | Videos |
|----------|----------|
| Non-Violence | 160 |
| Violence | 150 |
| Total | 310 |

---

# 3. Negative Mining Dataset Expansion

After evaluating the initial model, additional samples were added to the training dataset using a negative mining approach.

The goal was to improve classification performance and reduce false detections by introducing hard examples into the training process.

## Added Non-Violence Samples

| Source | Videos Added |
|----------|----------|
| AirtLab Dataset (CAM1) | 60 |

## Added Violence Samples

| Source | Videos Added |
|----------|----------|
| AirtLab Dataset (CAM2) | 15 |

### Source

- AirtLab Dataset for Automatic Violence Detection in Videos  
  https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos

---

# 4. Retraining and Evaluation

The model was retrained using the expanded training dataset containing the additional samples obtained through negative mining.

After retraining, the model was evaluated on the same testing dataset consisting of 310 videos.

The inclusion of additional hard examples improved the model's overall fight detection performance compared to the initial training stage.

---

# 5. Complete Dataset

The complete dataset used in this project is available at the following Google Drive link:

📂 **Download Dataset:**  
https://drive.google.com/drive/folders/194iU7PRBts5Z1jKzAMNRqSk1ds3h0GFE?usp=drive_link

> Note: All datasets included in this repository were collected from publicly available sources and were used solely for academic and research purposes. Proper credit belongs to the original dataset authors and maintainers.

---


