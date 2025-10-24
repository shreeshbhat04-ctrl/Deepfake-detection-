import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

# --- 1. DEFINE IMAGE TRANSFORMATIONS ---
# Images will be resized and normalized to match the ResNeXt input
IMG_SIZE = 224
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the face detector model you just downloaded
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# --- 2. PREPROCESSING FUNCTION ---
def extract_frames_from_video(video_path, sequence_length=20):
    """
    Processes a single video file.
    1. Opens video
    2. Reads frames
    3. Detects and crops the largest face
    4. Resizes, normalizes, and stacks frames into a tensor
    5. Selects `sequence_length` evenly spaced frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Error: Video {video_path} has no frames.")
        cap.release()
        return None
        
    processed_frames = []
    
    # Select `sequence_length` evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Find the largest face (by area w*h)
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            
            # Add some padding (10% on each side)
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            
            # Get coordinates, clamping to image boundaries
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(frame.shape[1] - 1, x + w + pad_w), min(frame.shape[0] - 1, y + h + pad_h)

            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                # Apply transforms (Resize, ToTensor, Normalize)
                processed_frame = data_transforms(face_crop)
                processed_frames.append(processed_frame)

    cap.release()

    if not processed_frames:
        print(f"Warning: No faces detected or processed in {video_path}")
        return None
    
    # If we couldn't get exactly `sequence_length` frames (e.g., no face in some),
    # duplicate the last valid frame to fill up
    while len(processed_frames) < sequence_length and len(processed_frames) > 0:
        processed_frames.append(processed_frames[-1])
        
    if not processed_frames:
        return None

    # Stack into a single tensor [T, C, H, W]
    return torch.stack(processed_frames)


# --- 3. PYTORCH DATASET CLASS ---
class DeepfakeDataset(Dataset):
    """
    A PyTorch Dataset class to load the deepfake videos.
    """
    def __init__(self, data_dir, sequence_length=20):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.video_files = []
        self.labels = []

        # Load REAL videos (label 0)
        real_path = os.path.join(data_dir, 'real')
        for filename in os.listdir(real_path):
            if filename.endswith('.mp4'):
                self.video_files.append(os.path.join(real_path, filename))
                self.labels.append(0)

        # Load FAKE videos (label 1)
        fake_path = os.path.join(data_dir, 'fake')
        for filename in os.listdir(fake_path):
            if filename.endswith('.mp4'):
                self.video_files.append(os.path.join(fake_path, filename))
                self.labels.append(1)
        
        print(f"Dataset found. Total videos: {len(self.video_files)}")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        
        # Process the video
        frames = extract_frames_from_video(video_path, self.sequence_length)
        
        # Handle cases where video processing failed
        if frames is None:
            # Return a dummy tensor and a special label (-1) to filter out
            print(f"Warning: Skipping {video_path}, processing failed.")
            return torch.zeros((self.sequence_length, 3, IMG_SIZE, IMG_SIZE)), -1 

        return frames, torch.tensor(label, dtype=torch.long)


# --- This code runs only if you execute `python dataset.py` directly ---
if __name__ == "__main__":
    # This is a test block to see if your dataset works.
    print("--- Running Dataset Test ---")
    
    # Assumes your data is in the 'data/' folder
    try:
        dataset = DeepfakeDataset(data_dir='data/')
        
        if len(dataset) == 0:
            print("\n!!! TEST FAILED: No videos found in 'data/real' or 'data/fake' folders.")
            print("Please add your .mp4 files to those folders and try again.")
        else:
            print(f"Successfully loaded {len(dataset)} videos.")
            
            # Try to get the first item
            frames, label = dataset[0]
            
            if label != -1:
                print("Successfully processed first video:")
                print(f"  Frames tensor shape: {frames.shape}")
                print(f"  Label: {label} (0=Real, 1=Fake)")
                print("\n--- TEST SUCCESSFUL ---")
            else:
                print("Could not process the first video. Check video file and warnings above.")
                print("--- TEST FAILED ---")

    except Exception as e:
        print(f"\n--- TEST FAILED ---")
        print(f"An error occurred: {e}")
        print("Did you create the 'data/real' and 'data/fake' folders?")
