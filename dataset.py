import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# Import MTCNN for better face detection
from mtcnn import MTCNN

# --- 1. CONFIGURATION ---
# 10 frames is enough for a resume project and runs faster on CPU
SEQUENCE_LENGTH_DEFAULT = 10 
IMG_SIZE = 224

# --- 2. INITIALIZE MTCNN ---
print(f"Initializing MTCNN...")
# FIX: The standard 'mtcnn' library doesn't take arguments here
mtcnn_detector = MTCNN()

# Standard normalization
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. PREPROCESSING FUNCTION ---
def extract_frames_from_video(video_path, sequence_length=SEQUENCE_LENGTH_DEFAULT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return None
        
    processed_frames = []
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue

        # Convert to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces
            faces = mtcnn_detector.detect_faces(frame_rgb)
            
            if len(faces) > 0:
                # Get highest confidence face
                best_face = sorted(faces, key=lambda x: x['confidence'], reverse=True)[0]
                x, y, w, h = best_face['box']
                
                # Fix negative coordinates
                x, y = max(0, x), max(0, y)
                # Add padding (10%)
                pad_w = int(w * 0.1)
                pad_h = int(h * 0.1)
                
                img_h, img_w, _ = frame.shape
                y1 = max(0, y - pad_h)
                y2 = min(img_h, y + h + pad_h)
                x1 = max(0, x - pad_w)
                x2 = min(img_w, x + w + pad_w)
                
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size != 0:
                    processed_frame = data_transforms(face_crop)
                    processed_frames.append(processed_frame)
        except Exception:
            continue

    cap.release()

    if not processed_frames:
        return None
    
    # Padding
    while len(processed_frames) < sequence_length:
        processed_frames.append(processed_frames[-1])

    return torch.stack(processed_frames[:sequence_length])


# --- 4. DATASET CLASS (With Limits) ---
class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, sequence_length=SEQUENCE_LENGTH_DEFAULT):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.video_files = []
        self.labels = []

        print(f"🔍 Scanning for videos in {data_dir}...")

        def find_videos_in_folder(folder_path):
            video_paths = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_paths.append(os.path.join(root, file))
            return video_paths

        # --- 1. REAL VIDEOS (Limit 400) ---
        real_path = os.path.join(data_dir, 'real')
        real_videos = find_videos_in_folder(real_path)
        
        if len(real_videos) > 400:
            print(f"   Found {len(real_videos)} REAL videos. Limiting to first 400.")
            real_videos = real_videos[:400]
        else:
            print(f"   Found {len(real_videos)} REAL videos.")

        for vid in real_videos:
            self.video_files.append(vid)
            self.labels.append(0)

        # --- 2. FAKE VIDEOS (Limit 400) ---
        fake_path = os.path.join(data_dir, 'fake')
        fake_videos = find_videos_in_folder(fake_path)
        
        if len(fake_videos) > 400:
            print(f"   Found {len(fake_videos)} FAKE videos. Limiting to first 400.")
            fake_videos = fake_videos[:400]
        else:
            print(f"   Found {len(fake_videos)} FAKE videos.")

        for vid in fake_videos:
            self.video_files.append(vid)
            self.labels.append(1)

        self.total_videos = len(self.video_files)
        print(f"✅ Total dataset size: {self.total_videos} videos")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        frames = extract_frames_from_video(video_path, self.sequence_length)
        
        if frames is None:
            return torch.zeros((self.sequence_length, 3, IMG_SIZE, IMG_SIZE)), -1 

        return frames, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    ds = DeepfakeDataset('data/')