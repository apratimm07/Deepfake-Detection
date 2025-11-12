# ==========================================================
#   DeepFake Detection — Full Image + Cropped Face + Grad-CAM
#   Author: Apratim Mishra, IIT Jammu
# ==========================================================

import os, glob, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
DATA_DIR   = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2"
MODEL_PATH = os.path.join(DATA_DIR, "deepfake_effb0_finetuned.pth")
DEMO_DIR   = os.path.join(DATA_DIR, "demo_images")  # full images here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️ Using device: {DEVICE}")

# ----------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()
print("✅ Model loaded successfully")

# ----------------------------------------------------------
# MTCNN FACE DETECTOR
# ----------------------------------------------------------
mtcnn = MTCNN(image_size=224, margin=10, post_process=True,
              device=DEVICE if DEVICE == "cuda" else None)

# ----------------------------------------------------------
# TRANSFORMS (same as training)
# ----------------------------------------------------------
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------------------------------------
# GRAD-CAM VISUALIZATION
# ----------------------------------------------------------
target_layers = [model.features[-1]]

def gradcam_visualize(face_crop, face_tensor):
    """Generate Grad-CAM heatmap for the given face crop."""
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=face_tensor,
                        targets=[BinaryClassifierOutputTarget(1)])[0, :]

    # ----- resize Grad-CAM to match face image size -----
    import cv2
    img_np = np.array(face_crop).astype(np.float32) / 255.0
    h, w, _ = img_np.shape
    grayscale_cam = cv2.resize(grayscale_cam, (w, h))     # 🔹 resize heatmap here

    # overlay
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    del cam  # prevent cleanup warnings
    return visualization


# ----------------------------------------------------------
# PREDICTION FUNCTION
# ----------------------------------------------------------
def predict_face(img_tensor):
    """Run model inference on a cropped face tensor."""
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).item()
    label = "FAKE 😈" if prob > 0.5 else "REAL 😇"
    conf  = prob if prob > 0.5 else 1 - prob
    return label, conf

# ----------------------------------------------------------
# IMAGE PIPELINE
# ----------------------------------------------------------
def predict_image(img_path):
    """Run full pipeline: detect face(s), classify, Grad-CAM visualize."""
    img = Image.open(img_path).convert("RGB")
    boxes, probs = mtcnn.detect(img)

    if boxes is None:
        print(f"⚠️ No face found in {os.path.basename(img_path)}")
        return

    # Copy full image with boxes
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for box in boxes:
        draw.rectangle(box.tolist(), outline="lime", width=3)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        face_crop = img.crop((x1, y1, x2, y2))
        face_tensor = tfm(face_crop).unsqueeze(0).to(DEVICE)

        # Prediction + Grad-CAM
        label, conf = predict_face(face_tensor)
        gradcam_img = gradcam_visualize(face_crop, face_tensor)

        # ---- Visualization ----
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(img_with_boxes)
        ax[0].set_title("Full Image")
        ax[0].axis("off")

        ax[1].imshow(np.array(face_crop))
        ax[1].set_title("Detected Face")
        ax[1].axis("off")

        ax[2].imshow(gradcam_img)
        ax[2].set_title("Grad-CAM Heatmap")
        ax[2].axis("off")

        plt.suptitle(
        f"{label} ({conf*100:.1f}%)",
        color="red" if label == "FAKE 😈" else "green",
        fontsize=13, fontweight="bold"
        )

        plt.tight_layout()
        plt.show()

        print(f"{os.path.basename(img_path)} [Face {i+1}] → {label} ({conf*100:.2f}%)")

# ----------------------------------------------------------
# RUN DEMO
# ----------------------------------------------------------
def run_demo(folder=DEMO_DIR):
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                  glob.glob(os.path.join(folder, "*.png")) +
                  glob.glob(os.path.join(folder, "*.jpeg")))
    if not imgs:
        print(f"⚠️ No images found in {folder}")
        return
    print(f"\n🔍 Running face-based inference on {len(imgs)} image(s)...\n")
    for p in imgs:
        predict_image(p)
    print("\n✅ Inference complete.")

# ----------------------------------------------------------
if __name__ == "__main__":
    run_demo()