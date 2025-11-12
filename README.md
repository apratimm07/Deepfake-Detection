# Deepfake-Detection

This project is part of the Computer Vision course at IIT Jammu. It presents a DeepFake Detection pipeline that integrates MTCNN for face detection, EfficientNet-B0 for binary classification, and Grad-CAM for visual explainability.
Our approach identifies subtle manipulation artifacts in facial regions, helping to differentiate synthetic (fake) and authentic (real) images. Frames are extracted from the Celeb-DF (v2) dataset.

# Workflow
Dataset (Celeb-DF-v2)
        ↓
Frame Extraction (Real + Synthetic Videos)
        ↓
Face Detection (MTCNN)
        ↓
Feature Extraction & Classification (EfficientNet-B0)
        ↓
Grad-CAM Visualization for Explainability

🧠 Model Components
1. MTCNN (Multi-task Cascaded Convolutional Networks)

Detects and aligns facial regions before feeding them to the classifier.

Composed of P-Net, R-Net, and O-Net, progressively refining face localization and landmark prediction.

2. EfficientNet-B0

Lightweight CNN architecture optimized via compound scaling of depth, width, and resolution.

Fine-tuned on cropped face regions (224×224 px) for binary classification (Real/Fake).

3. Grad-CAM (Gradient-Weighted Class Activation Mapping)

Generates heatmaps highlighting regions influencing the model’s decision.

Offers interpretability by showing where the network “looked” during inference.

📂 Dataset

Celeb-DF (v2) — a large-scale DeepFake dataset consisting of real and manipulated videos of celebrities and public figures.

Source: Kaggle – CelebDF-v2

465 paired videos (Real + Fake)

8 frames per video extracted (frame gap = 15)

Cropped and aligned faces using MTCNN → final 7438 balanced images

#Results

<img width="1263" height="472" alt="Screenshot 2025-11-12 235445" src="https://github.com/user-attachments/assets/25b34ac0-b754-45ec-a6d1-c2de122323a4" />
<img width="1667" height="581" alt="Screenshot 2025-11-12 233340" src="https://github.com/user-attachments/assets/2ef3cd46-407a-4698-98bc-82c5804f6898" />


