import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    """
    Extracts spatial features from a single frame using a pre-trained ResNeXt.
    """
    def __init__(self, freeze=True):
        super(FeatureExtractor, self).__init__()
        
        # Load a pretrained ResNeXt50
        # weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2 is the new syntax
        self.model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        
        # Freeze all layers in the network
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get the number of output features from the layer before the classifier
        # In ResNeXt, this is self.model.fc
        self.feature_dim = self.model.fc.in_features
        
        # Remove the final classification layer (we don't need 1000 ImageNet classes)
        # nn.Identity() is a placeholder that just passes the input through
        self.model.fc = nn.Identity() 

    def forward(self, x):
        # Input x has shape [B*T, C, H, W]
        # Output will have shape [B*T, feature_dim]
        return self.model(x)

class DeepfakeDetector(nn.Module):
    """
    Combines the CNN extractor and LSTM sequencer to classify a video.
    """
    def __init__(self, cnn_feature_dim, lstm_hidden_size=512, lstm_layers=2, num_classes=2, dropout=0.5):
        """
        Args:
            cnn_feature_dim (int): The output dimension from our FeatureExtractor (e.g., 2048 for ResNeXt50)
            lstm_hidden_size (int): The number of features in the LSTM's hidden state.
            lstm_layers (int): The number of stacked LSTM layers.
            num_classes (int): The number of output classes (2: Real/Fake).
            dropout (float): Dropout probability for regularization.
        """
        super(DeepfakeDetector, self).__init__()
        
        self.feature_extractor = FeatureExtractor(freeze=True)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        # --- Sequence Modeling (LSTM) ---
        # The LSTM will take the CNN features for each frame as input
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,       # Input shape is [BatchSize, SeqLength, Features]
            bidirectional=True,     # It will look at the sequence forwards and backwards
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # --- Classification Head ---
        # We'll build a small classifier on top of the LSTM's output
        self.fc1 = nn.Linear(
            lstm_hidden_size * 2,  # * 2 because the LSTM is bidirectional
            lstm_hidden_size // 2
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_classes) # Final output: 2 classes

    def forward(self, x):
        # Input x has shape: [B, T, C, H, W]
        # B = Batch Size
        # T = Sequence Length (e.g., 20 frames)
        # C, H, W = Frame dimensions (3, 224, 224)
        
        batch_size, seq_len, c, h, w = x.shape
        
        # --- 1. Feature Extraction (CNN) ---
        # We need to pass all frames through the CNN.
        # Reshape to [B * T, C, H, W] to treat all frames as one big batch.
        x_flat = x.view(batch_size * seq_len, c, h, w)
        
        features = self.feature_extractor(x_flat)
        # 'features' now has shape [B * T, cnn_feature_dim]
        
        # --- 2. Sequence Modeling (LSTM) ---
        # Reshape features back into sequences: [B, T, cnn_feature_dim]
        features_seq = features.view(batch_size, seq_len, -1)
        
        # Pass the sequence of features through the LSTM
        # lstm_out shape: [B, T, 2 * lstm_hidden_size] (because bidirectional)
        # h_n, c_n are the final hidden/cell states, which we don't need here
        lstm_out, (h_n, c_n) = self.lstm(features_seq)
        
        # We'll use the output from the *last* time step for classification
        # lstm_out[:, -1, :] gets the output of the last frame in the sequence
        last_time_step_out = lstm_out[:, -1, :]
        # Shape is now [B, 2 * lstm_hidden_size]
        
        # --- 3. Classification ---
        # Pass the LSTM's final output through our classifier
        x = self.dropout(self.relu(self.fc1(last_time_step_out)))
        out = self.fc2(x)
        # 'out' shape: [B, num_classes] (e.g., [8, 2])
        
        return out
