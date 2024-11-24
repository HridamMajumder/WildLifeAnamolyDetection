import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F

class APTv2Dataset(Dataset):

    ''' This class code was written by Mishra Aman Ashok '''

    def __init__(self, json_path: str, sequence_length: int = 15):
        """
        Dataset class for APTv2 data
        
        Args:
            json_path: Path to annotations JSON file
            sequence_length: Number of frames per sequence
        """
        with open(json_path) as f:
            self.data = json.load(f)
            
        self.sequence_length = sequence_length
        self.num_keypoints = 17  # From APTv2 format
        self.num_species = len(self.data['categories'])
        
        # Create image_id to image info mapping for quick access
        self.image_info = {img['id']: img for img in self.data['images']}
        
        # Create video_id to annotations mapping
        self.video_annotations = self._group_by_video()
        
        # Get valid sequences (videos with enough frames)
        self.valid_sequences = self._get_valid_sequences()
        
    def _group_by_video(self) -> Dict[int, List]:
        """Group annotations by video_id"""
        video_anns = {}
        for ann in self.data['annotations']:
            video_id = ann['video_id']
            if video_id not in video_anns:
                video_anns[video_id] = []
            video_anns[video_id].append(ann)
        
        # Sort annotations by image_id within each video
        for video_id in video_anns:
            video_anns[video_id].sort(key=lambda x: x['image_id'])
            
        return video_anns
    
    def _get_valid_sequences(self) -> List[Tuple[int, int]]:
        """Get list of valid (video_id, start_idx) pairs"""
        valid_sequences = []
        
        for video_id, anns in self.video_annotations.items():
            if len(anns) >= self.sequence_length:
                # Create sequences with overlap
                for i in range(0, len(anns) - self.sequence_length + 1):
                    valid_sequences.append((video_id, i))
                    
        return valid_sequences
    
    def _process_keypoints(self, 
                          keypoints: List[float], 
                          image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process raw keypoints into normalized coordinates
        
        Args:
            keypoints: List of keypoint coordinates and visibility
            image_id: ID of the image for getting dimensions
        """
        # Get image dimensions
        img_info = self.image_info[image_id]
        width = img_info['width']
        height = img_info['height']
        
        # Reshape to (num_keypoints, 3) - x, y, visibility
        kp = np.array(keypoints).reshape(-1, 3)
        
        # Get valid keypoints (visibility > 0)
        valid_mask = kp[:, 2] > 0
        
        # Normalize coordinates to [0, 1] using actual image dimensions
        kp[:, 0] = kp[:, 0] / width
        kp[:, 1] = kp[:, 1] / height
        
        return kp[:, :2], valid_mask
    
    def _extract_features(self, 
                         sequence: List[Dict]) -> np.ndarray:
        """Extract features from a sequence of annotations"""
        # Initialize arrays
        features = np.zeros((self.sequence_length, self.num_keypoints * 2))
        bbox_features = np.zeros((self.sequence_length, 4))
        
        for i, ann in enumerate(sequence):
            # Get image dimensions
            img_info = self.image_info[ann['image_id']]
            width = img_info['width']
            height = img_info['height']
            
            # Process keypoints
            kp, valid_mask = self._process_keypoints(ann['keypoints'], ann['image_id'])
            features[i] = kp.flatten()
            
            # Process bbox (normalize using actual image dimensions)
            bbox = np.array(ann['bbox'])
            bbox_features[i] = [
                bbox[0] / width,   # x
                bbox[1] / height,  # y
                bbox[2] / width,   # width
                bbox[3] / height   # height
            ]
            
        # Combine features
        combined_features = np.concatenate([
            features,
            bbox_features
        ], axis=1)
        
        return combined_features
    
    def __len__(self) -> int:
        return len(self.valid_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_id, start_idx = self.valid_sequences[idx]
        
        # Get sequence of annotations
        sequence = self.video_annotations[video_id][start_idx:start_idx + self.sequence_length]
        
        # Extract features using actual image dimensions
        features = self._extract_features(sequence)
        
        # Get category (species) ID
        category_id = sequence[0]['category_id']
        
        # Create one-hot encoding for species
        species_condition = np.zeros(self.num_species)
        species_condition[category_id] = 1
        
        return (
            torch.FloatTensor(features),
            torch.FloatTensor(species_condition)
        )


# The code from here onwards was written by Deep Vaghasiya


import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt


class ConditionalLSTMAutoencoder(nn.Module):



    def __init__(
        self,
        input_dim: int,          # 38 = 17*2 (keypoints) + 4 (bbox)
        num_classes: int,        # Number of species
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Condition embedding
        self.embed_class = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Combined input dimension
        self.combined_dim = input_dim + 16
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=self.combined_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Use bidirectional for better context
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim * 2 + 16,  # *2 for bidirectional
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the autoencoder
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]
            condition: Condition tensor of shape [batch_size, num_classes]
            
        Returns:
            tuple: (reconstructed_sequence, encoded_representation)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed condition
        condition_embedded = self.embed_class(condition)  # [batch_size, 16]
        condition_repeated = condition_embedded.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, 16]
        
        # Combine input with condition
        x_combined = torch.cat([x, condition_repeated], dim=-1)  # [batch_size, seq_len, input_dim + 16]
        
        # Encode
        encoded_seq, (hidden, cell) = self.encoder(x_combined)
        # encoded_seq shape: [batch_size, seq_len, hidden_dim * 2] (due to bidirectional)
        
        # Combine encoded sequence with condition for decoder
        decoder_input = torch.cat([encoded_seq, condition_repeated], dim=-1)
        # decoder_input shape: [batch_size, seq_len, hidden_dim * 2 + 16]
        
        # Decode
        decoded_seq, _ = self.decoder(decoder_input)
        # decoded_seq shape: [batch_size, seq_len, hidden_dim]
        
        # Generate output
        output = self.output_layer(decoded_seq)
        # output shape: [batch_size, seq_len, input_dim]
        
        return output, encoded_seq

class AnomalyDetector:
    def __init__(self, num_keypoints: int = 17):
        self.num_keypoints = num_keypoints
        
    def compute_reconstruction_error(self, 
                                   original: torch.Tensor, 
                                   reconstructed: torch.Tensor,
                                   return_components: bool = False
                                  ) -> torch.Tensor:
        # Separate pose and bbox components
        original_pose = original[..., :-4]
        original_bbox = original[..., -4:]
        reconstructed_pose = reconstructed[..., :-4]
        reconstructed_bbox = reconstructed[..., -4:]
        
        # Reshape pose for keypoint-wise error
        original_pose = original_pose.view(*original_pose.shape[:-1], self.num_keypoints, 2)
        reconstructed_pose = reconstructed_pose.view(*reconstructed_pose.shape[:-1], self.num_keypoints, 2)
        
        # Compute pose error (Euclidean distance per keypoint)
        pose_error = torch.sqrt(((reconstructed_pose - original_pose) ** 2).sum(dim=-1))
        
        # Compute bbox error
        bbox_error = F.mse_loss(reconstructed_bbox, original_bbox, reduction='none')
        
        # Compute temporal consistency
        pose_velocity_error = F.mse_loss(
            reconstructed[..., :-4][:, 1:] - reconstructed[..., :-4][:, :-1],
            original[..., :-4][:, 1:] - original[..., :-4][:, :-1],
            reduction='none'
        )
        
        # Combine errors with weights
        total_error = (
            pose_error.mean(dim=-1) * 1.0 +  # Average over keypoints
            bbox_error.mean(dim=-1) * 0.5 +  # Average over bbox components
            F.pad(pose_velocity_error.mean(dim=-1), (0, 1)) * 0.5  # Add padding for last frame
        )
        
        if return_components:
            return {
                'total_error': total_error,
                'pose_error': pose_error,
                'bbox_error': bbox_error,
                'velocity_error': pose_velocity_error
            }
        
        return total_error

def train_model(
    model: ConditionalLSTMAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.0001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    anomaly_detector = AnomalyDetector()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_features, batch_conditions in train_loader:
            batch_features = batch_features.to(device)
            batch_conditions = batch_conditions.to(device)
            
            optimizer.zero_grad()
            
            reconstructed, _ = model(batch_features, batch_conditions)
            
            error_components = anomaly_detector.compute_reconstruction_error(
                batch_features,
                reconstructed,
                return_components=True
            )
            
            loss = error_components['total_error'].mean()
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_errors = []
        
        with torch.no_grad():
            for batch_features, batch_conditions in val_loader:
                batch_features = batch_features.to(device)
                batch_conditions = batch_conditions.to(device)
                
                reconstructed, _ = model(batch_features, batch_conditions)
                error_components = anomaly_detector.compute_reconstruction_error(
                    batch_features,
                    reconstructed,
                    return_components=True
                )
                
                val_loss = error_components['total_error'].mean()
                total_val_loss += val_loss.item()
                val_errors.extend(error_components['total_error'].cpu().numpy().flatten())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        # Calculate anomaly threshold
        anomaly_threshold = np.percentile(val_errors, 95)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Anomaly Threshold: {anomaly_threshold:.4f}')
        print('-' * 50)
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Plot results
    plot_training_results(train_losses, val_losses, val_errors, anomaly_threshold)
    
    return model, train_losses, val_losses, anomaly_threshold

def plot_training_results(train_losses, val_losses, val_errors, anomaly_threshold):
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Error distribution plot
    plt.subplot(1, 2, 2)
    plt.hist(val_errors, bins=50, density=True)
    plt.axvline(x=anomaly_threshold, color='r', linestyle='--', label='Anomaly Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Validation Error Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def detect_anomalies(
    model: ConditionalLSTMAutoencoder,
    test_loader: DataLoader,
    anomaly_threshold: float,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    anomaly_detector = AnomalyDetector()
    
    all_errors = []
    all_anomalies = []
    
    with torch.no_grad():
        for batch_features, batch_conditions in test_loader:
            batch_features = batch_features.to(device)
            batch_conditions = batch_conditions.to(device)
            
            reconstructed, _ = model(batch_features, batch_conditions)
            error_components = anomaly_detector.compute_reconstruction_error(
                batch_features,
                reconstructed,
                return_components=True
            )
            
            errors = error_components['total_error']
            anomalies = errors > anomaly_threshold
            
            all_errors.append(errors.cpu().numpy())
            all_anomalies.append(anomalies.cpu().numpy())
    
    return np.concatenate(all_errors), np.concatenate(all_anomalies)

def main(train_json_path: str, val_json_path: str, test_json_path: str):
    # Create datasets
    train_dataset = APTv2Dataset(train_json_path)
    val_dataset = APTv2Dataset(val_json_path)
    test_dataset = APTv2Dataset(test_json_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = ConditionalLSTMAutoencoder(
        input_dim=38,  # 17*2 (keypoints) + 4 (bbox)
        num_classes=train_dataset.num_species,
        hidden_dim=128
    )
    
    # Train model
    model, train_losses, val_losses, anomaly_threshold = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=50,
        learning_rate=0.0001
    )
    
    # Detect anomalies in test set
    test_errors, test_anomalies = detect_anomalies(model, test_loader, anomaly_threshold)
    
    # Calculate and print results
    anomaly_ratio = test_anomalies.mean()
    print(f"\nTest Set Results:")
    print(f"Anomaly Ratio: {anomaly_ratio:.4f}")
    print(f"Mean Error: {test_errors.mean():.4f}")
    print(f"Error Std: {test_errors.std():.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'anomaly_threshold': anomaly_threshold,
        'test_anomaly_ratio': anomaly_ratio,
        'test_errors': test_errors,
        'test_anomalies': test_anomalies
    }

if __name__ == "__main__":
    train_json_path = '/home/deepb/Desktop/AIDS/project/APTv2_annotations/APTv2/annotations/train_annotations.json'
    val_json_path = '/home/deepb/Desktop/AIDS/project/APTv2_annotations/APTv2/annotations/val_annotations.json'
    test_json_path = '/home/deepb/Desktop/AIDS/project/APTv2_annotations/APTv2/annotations/test_annotations_hard.json'
    
    results = main(train_json_path, val_json_path, test_json_path)