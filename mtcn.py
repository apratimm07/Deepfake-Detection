# ========================================
#   MTCNN Face Cropping (CPU or GPU)
# ========================================

from facenet_pytorch import MTCNN
from PIL import Image
from pathlib import Path
import os, glob
from tqdm import tqdm
import torch

# --- source folders (your frame outputs) ---
SRC_REAL = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\image_real"
SRC_FAKE = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\image_synthetic"

# --- destination folders (cropped faces) ---
DST_REAL = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\faces_real"
DST_FAKE = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\faces_synthetic"

os.makedirs(DST_REAL, exist_ok=True)
os.makedirs(DST_FAKE, exist_ok=True)

# --- choose device (auto) ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"⚙️ Using device: {device}")

# --- initialize MTCNN ---
mtcnn = MTCNN(image_size=224, margin=10, post_process=True, device=device)

def crop_faces(src_dir, dst_dir):
    """Crop faces from all images in src_dir and save to dst_dir."""
    images = sorted(glob.glob(f"{src_dir}/*.jpg"))
    for img_path in tqdm(images, desc=f"Cropping {Path(src_dir).name}", ncols=100):
        try:
            img = Image.open(img_path).convert("RGB")
            out_path = Path(dst_dir) / Path(img_path).name
            mtcnn(img, save_path=str(out_path))
        except Exception as e:
            # Skip if no face detected or invalid image
            continue

# --- process both real and fake folders ---
crop_faces(SRC_REAL, DST_REAL)
crop_faces(SRC_FAKE, DST_FAKE)

print("✅ Face cropping complete!")
print(f"Saved in:\n{DST_REAL}\n{DST_FAKE}")
