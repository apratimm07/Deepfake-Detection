import cv2, os, glob
from pathlib import Path

# === Input paths ===
REAL_VID_DIR = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\Celeb-real"
FAKE_VID_DIR = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\Celeb-synthesis"

# === Output paths ===
OUT_REAL = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\image_real"
OUT_FAKE = r"C:\Users\Apratim\cs\Deepfake\Celeb-DF-v2\image_synthetic"

os.makedirs(OUT_REAL, exist_ok=True)
os.makedirs(OUT_FAKE, exist_ok=True)

# === Extraction configuration ===
STEP = 15        # frame gap
MAX_FRAMES = 8   # number of frames to extract per video

def extract_equal_frames(real_path, fake_path, real_out, fake_out, index_start):
    """Extract equal number of frames from real & fake videos (in parallel)."""
    cap_real = cv2.VideoCapture(real_path)
    cap_fake = cv2.VideoCapture(fake_path)

    frame_idx, saved = 0, 0
    while True:
        ret_r, frame_r = cap_real.read()
        ret_f, frame_f = cap_fake.read()
        if not ret_r or not ret_f:
            break
        if frame_idx % STEP == 0 and saved < MAX_FRAMES:
            real_name = f"real_{index_start + saved:06d}.jpg"
            fake_name = f"synthetic_{index_start + saved:06d}.jpg"

            cv2.imwrite(os.path.join(real_out, real_name), frame_r)
            cv2.imwrite(os.path.join(fake_out, fake_name), frame_f)
            saved += 1
        if saved >= MAX_FRAMES:
            break
        frame_idx += 1

    cap_real.release()
    cap_fake.release()
    return index_start + saved


# === Main script ===
real_videos = sorted(glob.glob(f"{REAL_VID_DIR}/*.mp4"))
fake_videos = sorted(glob.glob(f"{FAKE_VID_DIR}/*.mp4"))

# Ensure both lists are aligned
num_pairs = min(len(real_videos), len(fake_videos))
print(f"✅ Found {num_pairs} pairs of real+fake videos")

counter = 0
for i in range(num_pairs):
    rvid, fvid = real_videos[i], fake_videos[i]
    counter = extract_equal_frames(rvid, fvid, OUT_REAL, OUT_FAKE, counter)

print(f"✅ Extraction complete: {counter} pairs of frames saved for both real & synthetic.")
