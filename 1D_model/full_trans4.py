import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import torch.nn.functional as F
import os
from torch.optim.lr_scheduler import StepLR
import random
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt



torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

#  GPU 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_bands, latent_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, num_bands))
        nn.init.xavier_uniform_(self.positional_encoding)
    def forward(self, x):
        return x + self.positional_encoding

class HybridBandSelectionModel(nn.Module):
    def __init__(self, num_bands, latent_dim, num_heads, num_layers, num_selected_bands):
        super(HybridBandSelectionModel, self).__init__()
        self.num_selected_bands = num_selected_bands

        self.band_importance = nn.Parameter(torch.zeros(num_bands))
       
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_bands, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=0.15
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize Transformer weights
        self._init_transformer_weights()

        # Positional Encoding 
        self.positional_encoding = LearnablePositionalEncoding(num_bands, latent_dim)

        # Lightweight Encoder
        self.lightweight_encoder = nn.Sequential(
            nn.Linear(num_selected_bands, 64),
            #nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.Sigmoid()
            
        )
        # Initialize Lightweight Encoder weights
        self._init_linear_weights(self.lightweight_encoder)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            #nn.Linear(32, 64),
            #nn.ReLU(),
            nn.Linear(64, num_bands),
            nn.Sigmoid()
        
        )
        
        # Initialize Decoder weights
        self._init_linear_weights(self.decoder)

    def _init_transformer_weights(self):
        """Initialize Transformer weights with Xavier/Glorot initialization."""
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_linear_weights(self, sequential_model):
        """Initialize Linear layer weights with Xavier/Glorot initialization."""
        for layer in sequential_model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


    def forward(self, x, x_original):
        batch_size, num_bands, height, width = x.shape

        # Flatten spatial dimensions
        x = x.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_bands)  # Shape: (batch_size * height * width, num_bands)

        # Transformer Encoder (with positional encoding)
        x = self.positional_encoding(x.unsqueeze(0)).squeeze(0)  # Add positional encoding
        encoded_bands = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)  # Shape: (batch_size * height * width, num_bands)

        attention_scores = encoded_bands * torch.sigmoid(self.band_importance)

        # Select top-k bands based on attention scores
        top_k_scores, top_k_indices = torch.topk(attention_scores, k=self.num_selected_bands, dim=1)  # Shape: (batch_size * height * width, num_selected_bands)

        # Sort the selected bands by their indices
        sorted_indices = torch.argsort(top_k_indices, dim=1)  # Shape: (batch_size * height * width, num_selected_bands)
        sorted_top_k_indices = torch.gather(top_k_indices, dim=1, index=sorted_indices)  # Shape: (batch_size * height * width, num_selected_bands)
        sorted_top_k_scores = torch.gather(top_k_scores, dim=1, index=sorted_indices)  # Shape: (batch_size * height * width, num_selected_bands)

        # Crop selected bands from the original input
        selected_bands1 = torch.gather(x_original.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_bands), dim=1, index=sorted_top_k_indices)  # Shape: (batch_size * height * width, num_selected_bands)

        # Multiply cropped bands by their attention scores
        selected_bands = selected_bands1 * sorted_top_k_scores  # Shape: (batch_size * height * width, num_selected_bands)

        # Lightweight Encoder 
        encoded_selected_bands = self.lightweight_encoder(selected_bands)  # Shape: (batch_size * height * width, latent_dim)

        # Decoder (reconstruct full spectrum)
        reconstructed = self.decoder(encoded_selected_bands)  # Shape: (batch_size * height * width, num_bands)

        # Reshape output back to original spatial dimensions
        reconstructed = reconstructed.reshape(batch_size, height, width, num_bands).permute(0, 3, 1, 2)  # Shape: (batch_size, num_bands, height, width)

        return reconstructed, sorted_top_k_indices, attention_scores, selected_bands1 


def attention_diversity_loss(attention_scores, penalty_strength=1.0):
    """
    Directly encourages diversity in attention scores across bands
    """
    # Compute mean attention across spatial dimensions
    mean_attn = attention_scores.mean(dim=0)  # shape [num_bands]
    
    # Add small epsilon and normalize
    norm_attn = (mean_attn + 1e-12) / (mean_attn.sum() + 1e-12 * mean_attn.size(0))
    
    # Compute entropy with clipping for numerical stability
    log_attn = torch.log(torch.clamp(norm_attn, min=1e-12, max=1.0))
    entropy = -torch.sum(norm_attn * log_attn)
    
    # Compute max entropy (uniform distribution)
    max_entropy = torch.log(torch.tensor(mean_attn.size(0), 
                                       dtype=torch.float32,
                                       device=mean_attn.device))
    
    # Loss is the difference from max entropy
    return penalty_strength * (max_entropy - entropy)


#20 #0.09
#def loss_function(reconstructed, target, indices, attention_scores, lambda_reg=0.009, lambda_div=20):
def loss_function(reconstructed, target, indices, attention_scores, selected, lambda_reg=10, lambda_div=0.001):
    reconstruction_loss = F.l1_loss(reconstructed, target)
    #diversity_loss = attention_diversity_loss(attention_scores)
    regularization_loss = torch.mean(selected**2)
    #diversity_loss = diversity_penalty(indices, penalty_strength=1)
    # Total loss
    total_loss = 10*reconstruction_loss  + lambda_reg * regularization_loss #+ lambda_div * diversity_loss
    return total_loss, reconstruction_loss,  regularization_loss#, diversity_loss,

class SpectralDataset(Dataset):
    def __init__(self, file_paths, patch_size, start_band=28, save_norm_path="norm_data.npz", augment=True):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.start_band = start_band
        self.augment = augment

        # Compute the maximum height and width in the dataset
        self.max_height = max([np.load(fp).shape[0] for fp in file_paths])
        self.max_width = max([np.load(fp).shape[1] for fp in file_paths])

        # Ensure the padded size is divisible by patch_size
        self.padded_height = ((self.max_height - 1) // patch_size + 1) * patch_size
        self.padded_width = ((self.max_width - 1) // patch_size + 1) * patch_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy file
        image = np.load(self.file_paths[idx])  # Shape: (height, width, num_bands)
        image = image[:, :, self.start_band:]  # Select bands
        image = image[:, :, :400]
    

        # ====== AUGMENTATION ======
        if self.augment:
            # Random flip (50% chance) 
            if random.random() < 0.5:
                # Flip left-right (width dimension)
                image = np.flip(image, axis=1)
            
            if random.random() < 0.5:
                # Flip up-down (height dimension)
                image = np.flip(image, axis=0)
                
            # Random rotation 
            if random.random() < 0.5:
                image = np.rot90(image, k=2, axes=(0, 1))  # 180 degrees
        # ====== END AUGMENTATION ======

        # Pad the image to the common size
        h, w, _ = image.shape
        pad_h = self.padded_height - h
        pad_w = self.padded_width - w
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')  # Shape: (padded_height, padded_width, num_bands - start_band)

        # Extract patches
        patches = []
        for i in range(0, self.padded_height, self.patch_size):
            for j in range(0, self.padded_width, self.patch_size):
                patch = padded_image[i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
        patches = np.stack(patches)  # Shape: (num_patches, patch_size, patch_size, num_bands - start_band)

        # Convert to tensor
        patches = torch.tensor(patches, dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: (num_patches, num_bands - start_band, patch_size, patch_size)
        return patches

def custom_collate_fn(batch):
    # Concatenate all patches into a single tensor
    patches = torch.cat(batch, dim=0)  # Shape: (total_patches, num_bands - start_band, patch_size, patch_size)
    return patches


num_bands = 400 #400  # Number of spectral bands
latent_dim =8   # Latent dimension
num_heads = 8  # Number of attention heads
num_layers = 2  # 3 Number of transformer layers
num_selected_bands = 10  # Number of selected bands
patch_size = 16  # Patch size
batch_size = 1  # Batch size
num_epochs = 100  # Number of epochs
learning_rate = 1e-3  # Learning rate

# Load data
data_dir = "/workspace/src/Season_4/Normal/sample_cubes"
file_paths = glob.glob(f"{data_dir}/*.npy")
dataset = SpectralDataset(file_paths, patch_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Initialize model, optimizer, and loss function
model = HybridBandSelectionModel(num_bands, latent_dim, num_heads, num_layers, num_selected_bands).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for batch in dataloader:
        # Move batch to GPU
        batch = batch.to(device)

        # Forward pass
        reconstructed, indices, attention_scores, selected = model(batch, batch)

        # Compute loss
        #loss, recon, div, reg = loss_function(reconstructed, batch, indices, attention_scores, selected)
        loss, recon, reg = loss_function(reconstructed, batch, indices, attention_scores, selected)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step the scheduler
    scheduler.step()

    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, recon: {recon.item():.4f}, reg: {reg.item():.4f}, div: {div.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, recon: {recon.item():.4f}, reg: {reg.item():.4f}")
    print(indices)
    #print(attention_scores)
torch.save(model.state_dict(), "new_former_10.pth")