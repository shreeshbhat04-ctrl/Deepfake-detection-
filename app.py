import streamlit as st
import torch
import os
import tempfile
import cv2
import numpy as np

# Import our custom modules
# We need these to rebuild the model and process the video
from model import DeepfakeDetector, FeatureExtractor
from dataset import extract_frames_from_video, IMG_SIZE # We reuse this function

# --- Configuration ---
SAVED_MODEL_PATH = 'saved_models/deepfake_detector_best.pth'
SEQUENCE_LENGTH = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Load the Trained Model ---
@st.cache_resource # Caches the model so it only loads once
def load_model():
    print("Loading model...")
    # Get the feature dimension from the CNN
    temp_cnn = FeatureExtractor(freeze=True)
    FEATURE_DIM = temp_cnn.feature_dim
    del temp_cnn

    # Instantiate the model with the same architecture as in train.py
    model = DeepfakeDetector(
        cnn_feature_dim=FEATURE_DIM,
        lstm_hidden_size=512,
        lstm_layers=2
    ).to(device)

    # Load the saved weights
    try:
        model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=device))
    except FileNotFoundError:
        st.error(f"Model file not found at {SAVED_MODEL_PATH}. Did you run train.py?")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    model.eval() # Set model to evaluation mode (very important!)
    print("Model loaded successfully.")
    return model

model = load_model()

# --- 2. Build the Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Deepfake Video Detector")
st.markdown("Upload a video to check if it's a 'REAL' or 'FAKE' (AI-generated) video.")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
    
    # Placeholder for results
    result_placeholder = st.empty()

with col2:
    st.header("Video Preview")
    video_placeholder = st.empty()


# --- 3. Run Inference when File is Uploaded ---
if uploaded_file is not None and model is not None:
    
    # Save the uploaded file to a temporary location
    # This is necessary because cv2.VideoCapture needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    # Display the video
    video_placeholder.video(temp_video_path)

    # Process the video and run the model
    with st.spinner('Analyzing video... This may take a moment.'):
        try:
            # Reuse our processing function from dataset.py
            frames_tensor = extract_frames_from_video(
                video_path=temp_video_path,
                face_cascade_path='haarcascade_frontalface_default.xml',
                sequence_length=SEQUENCE_LENGTH,
                img_size=IMG_SIZE
            )

            if frames_tensor is None:
                result_placeholder.error("Could not detect a face in the video. Please try another video.")
            else:
                # Add a batch dimension and send to device
                # [T, C, H, W] -> [1, T, C, H, W]
                frames_tensor = frames_tensor.unsqueeze(0).to(device)

                # Run inference
                with torch.no_grad():
                    output = model(frames_tensor)
                    _, pred = torch.max(output, 1)
                    prediction = pred.item()

                # Display the result
                if prediction == 1:
                    result_placeholder.error("Prediction: FAKE 🟥", icon="🚨")
                else:
                    result_placeholder.success("Prediction: REAL 🟩", icon="✅")

        except Exception as e:
            result_placeholder.error(f"An error occurred during analysis: {e}")
        
        # Clean up the temporary file
        os.remove(temp_video_path)
