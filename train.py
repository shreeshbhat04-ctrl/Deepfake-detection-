import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import os

# Import our custom modules
from dataset import DeepfakeDataset
from model import DeepfakeDetector, FeatureExtractor

# --- 1. Configuration ---
DATA_DIR = 'data/'
SAVED_MODEL_PATH = 'saved_models/deepfake_detector_best.pth'
SEQUENCE_LENGTH = 20
BATCH_SIZE = 4       # Lower this if you get "CUDA out of memory" errors
EPOCHS = 10          # Start with 10-20, increase for better results
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.25 # Use 25% of data for validation (e.g., 9 train, 3 val for 12 videos)

def collate_fn(batch):
    """
    Filters out failed video processing (where label is -1)
    """
    batch = list(filter(lambda x: x[1] != -1, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    # --- 2. Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # --- 3. Load Dataset ---
    print("Loading dataset...")
    full_dataset = DeepfakeDataset(data_dir=DATA_DIR, sequence_length=SEQUENCE_LENGTH)
    
    if len(full_dataset) == 0:
        print("Error: No data found. Please check your 'data/' folder.")
        return

    # Split dataset into training and validation
    total_size = len(full_dataset)
    val_size = int(total_size * VALIDATION_SPLIT)
    train_size = total_size - val_size
    
    # Ensure val_size is at least 1 if dataset is small
    if val_size == 0 and train_size > 0:
        val_size = 1
        train_size = total_size - 1

    if train_size == 0 or val_size == 0:
        print(f"Dataset is too small to split. Found only {total_size} videos. Need at least 2.")
        return

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Total videos: {total_size} | Training: {train_size} | Validation: {val_size}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for Windows/Jupyter compatibility
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )

    # --- 4. Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    # Get the feature dimension from the CNN
    temp_cnn = FeatureExtractor(freeze=True)
    FEATURE_DIM = temp_cnn.feature_dim
    del temp_cnn
    
    model = DeepfakeDetector(
        cnn_feature_dim=FEATURE_DIM,
        lstm_hidden_size=512,
        lstm_layers=2
    ).to(device)

    # We only want to train the unfrozen weights (LSTM and classifier)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    print(f"Found {len(params_to_update)} trainable parameter tensors.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE)

    # --- 5. Training Loop ---
    print("--- Starting Training ---")
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        
        for i, (videos, labels) in enumerate(train_loader):
            if videos.nelement() == 0: # Skip empty batches (from collate_fn)
                continue
                
            videos, labels = videos.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            
            if len(train_loader) > 0:
                 print(f'  Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}')


        avg_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for videos, labels in val_loader:
                if videos.nelement() == 0: continue
                
                videos, labels = videos.to(device), labels.to(device)
                outputs = model(videos)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        val_accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        
        print(f'\nEpoch {epoch+1}/{EPOCHS} Summary:')
        print(f'  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy*100:.2f}%')

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(SAVED_MODEL_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVED_MODEL_PATH)
            print(f'  *** New best model saved to {SAVED_MODEL_PATH} (Accuracy: {val_accuracy*100:.2f}%) ***\n')

    print(f"--- Training Finished ---")
    print(f"Best validation accuracy: {best_val_accuracy*100:.2f}%")

if __name__ == "__main__":
    main()
