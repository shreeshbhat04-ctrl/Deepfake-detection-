from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import shutil
import tempfile
import torch.nn.functional as F

# Import your custom modules
from model import DeepfakeDetector, FeatureExtractor
from dataset import extract_frames_from_video

# --- 1. CONFIGURATION ---
SAVED_MODEL_PATH = 'saved_models/deepfake_detector_best.pth'
SEQUENCE_LENGTH = 10
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

# --- 2. CORS SETUP (Crucial for TypeScript Frontend) ---
# This allows your frontend (usually localhost:3000) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD MODEL AT STARTUP ---
# We load this once so we don't have to reload it for every request
print("Loading AI Model...")
temp_cnn = FeatureExtractor(freeze=True)
feature_dim = temp_cnn.feature_dim
del temp_cnn

model = DeepfakeDetector(
    cnn_feature_dim=feature_dim,
    lstm_hidden_size=512,
    lstm_layers=2
).to(DEVICE)

try:
    model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise RuntimeError("Model failed to load")


# --- 4. THE PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    """
    Endpoint to receive a video file and return {label: 'REAL'/'FAKE', confidence: float}
    """
    
    # 1. Validate file type
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload .mp4, .mov, or .avi")

    # 2. Save uploaded file to a temporary file on disk
    # OpenCV needs a file path, it can't read from memory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # 3. Preprocess the video (Extract Faces)
        # Using your logic from dataset.py
        frames_tensor = extract_frames_from_video(
            video_path=temp_file_path,
            sequence_length=SEQUENCE_LENGTH
        )

        if frames_tensor is None:
            # 202 is "Accepted" but processing failed logic
            return {"status": "error", "message": "Could not detect a face in the video."}

        # 4. Run Inference
        frames_tensor = frames_tensor.unsqueeze(0).to(DEVICE) # Add batch dimension

        with torch.no_grad():
            output = model(frames_tensor)
            
            # Calculate percentages
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Label 0 = REAL, Label 1 = FAKE (Based on your dataset.py logic)
            prediction_idx = predicted_class.item()
            conf_score = confidence.item() * 100

            result_label = "FAKE" if prediction_idx == 1 else "REAL"

        # 5. Return JSON response
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result_label,
            "confidence": round(conf_score, 2),
            "is_fake": prediction_idx == 1
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 6. Cleanup: Delete the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

