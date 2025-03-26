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


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if GPU is available
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
       
        #nn.init.normal_(self.band_importance)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_bands, 
            nhead=num_heads, 
            dim_feedforward=128, 
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Initialize Transformer weights
        self._init_transformer_weights()

        # Positional Encoding (already initialized)
        self.positional_encoding = LearnablePositionalEncoding(num_bands, latent_dim)

        # Lightweight Encoder
        self.lightweight_encoder = nn.Sequential(
            nn.Linear(num_selected_bands, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # Initialize Lightweight Encoder weights
        self._init_linear_weights(self.lightweight_encoder)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_bands)
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
        # Compute attention scores (mean across heads)
        #attention_scores = encoded_bands  # Shape: (batch_size * height * width, num_bands)

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

        # Lightweight Encoder (encode cropped bands into latent space)
        encoded_selected_bands = self.lightweight_encoder(selected_bands)  # Shape: (batch_size * height * width, latent_dim)

        # Decoder (reconstruct full spectrum)
        reconstructed = self.decoder(encoded_selected_bands)  # Shape: (batch_size * height * width, num_bands)

        # Reshape output back to original spatial dimensions
        reconstructed = reconstructed.reshape(batch_size, height, width, num_bands).permute(0, 3, 1, 2)  # Shape: (batch_size, num_bands, height, width)

        return reconstructed, sorted_top_k_indices, attention_scores, selected_bands1 

def diversity_penalty(selected_indices, penalty_strength=1):
    """
    Penalizes the selection of bands that are close to each other.
    """
    diff = torch.diff(selected_indices, dim=1)  # Compute differences between adjacent bands
    penalty = torch.mean(torch.exp(-penalty_strength * torch.abs(diff)))  # Normalized penalty
    return penalty

def spectral_angle_loss(reconstructed, target, eps=1e-10):
    # Normalize the vectors
    reconstructed_norm = reconstructed / (reconstructed.norm(dim=-1, keepdim=True) + eps)
    target_norm = target / (target.norm(dim=-1, keepdim=True) + eps)
    
    # Compute cosine similarity and convert to loss (1 - cosθ)
    cos_theta = (reconstructed_norm * target_norm).sum(dim=-1)
    sal_loss = 1.0 - cos_theta  # Range [0, 2] where 0 = perfect match
    
    return torch.mean(sal_loss)  # Average over batch

#20 #0.09
#def loss_function(reconstructed, target, indices, attention_scores, lambda_reg=0.009, lambda_div=20):
def loss_function(reconstructed, target, indices, attention_scores, selected, lambda_reg=0.1, lambda_div=1):
    # Reconstruction loss (MSE)
    #reconstruction_loss = F.mse_loss(reconstructed, target)
    reconstruction_loss = F.l1_loss(reconstructed, target)
    spectral_loss = spectral_angle_loss(reconstructed, target, eps=1e-10)

    # Regularization term (encourage sparsity in attention scores)
    #regularization_loss = torch.mean(torch.abs(attention_scores))/ (num_bands * batch_size)
    regularization_loss = torch.mean(selected**2)
    diversity_loss = diversity_penalty(indices, penalty_strength=1)
    # Total loss
    total_loss = reconstruction_loss + lambda_reg * regularization_loss #+ lambda_div * diversity_loss
    return total_loss, reconstruction_loss, diversity_loss, regularization_loss

class SpectralDataset(Dataset):
    def __init__(self, file_paths, patch_size, start_band=48, save_norm_path="norm_data.npz"):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.start_band = start_band  # Start from the 50th band

        # Compute the maximum height and width in the dataset
        self.max_height = max([np.load(fp).shape[0] for fp in file_paths])
        self.max_width = max([np.load(fp).shape[1] for fp in file_paths])

        # Ensure the padded size is divisible by patch_size
        self.padded_height = ((self.max_height - 1) // patch_size + 1) * patch_size
        self.padded_width = ((self.max_width - 1) // patch_size + 1) * patch_size

        # Compute mean and std for normalization (only on non-padded regions)
        #self.mean, self.std = self.compute_mean_std()
        # Save normalization parameters if a path is provided
        #np.savez(save_norm_path, mean=self.mean, std=self.std)
        #print(f"Normalization data saved to {save_norm_path}")
        # Compute min and max for normalization (only on non-zero regions)
        #self.min, self.max = self.compute_min_max()

        # Save normalization parameters if a path is provided
        #np.savez(save_norm_path, min=self.min, max=self.max)
        print(f"Normalization data saved to {save_norm_path}")
        

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load .npy file
        image = np.load(self.file_paths[idx])  # Shape: (height, width, num_bands)

        # Use only bands from the 50th band onwards
        image = image[:, :, self.start_band:]  # Shape: (height, width, num_bands - start_band)

        # Normalize the image
        #image = (image - self.mean) / self.std
        #image = (image - self.min) / (self.max - self.min)

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

    # def compute_mean_std(self):
    #     # Compute mean and std for normalization (only on non-zero regions)
    #     all_data = []
    #     for fp in self.file_paths:
    #         image = np.load(fp)[:, :, self.start_band:]  # Shape: (height, width, num_bands - start_band)
            
    #         # Flatten the image and exclude zero regions
    #         flattened_image = image.reshape(-1, image.shape[-1])  # Shape: (height * width, num_bands - start_band)
    #         non_zero_mask = np.any(flattened_image != 0, axis=1)  # Mask for non-zero pixels
    #         non_zero_pixels = flattened_image[non_zero_mask]  # Shape: (num_non_zero_pixels, num_bands - start_band)
            
    #         all_data.append(non_zero_pixels)

    #     # Concatenate all non-zero pixels along the first dimension
    #     all_data = np.concatenate(all_data, axis=0)  # Shape: (total_non_zero_pixels, num_bands - start_band)

    #     # Compute mean and std over all non-zero pixels
    #     mean = np.mean(all_data, axis=0)  # Shape: (num_bands - start_band,)
    #     std = np.std(all_data, axis=0)  # Shape: (num_bands - start_band,)
    #     return mean, std

    def compute_min_max(self):
        # Compute min and max for normalization (only on non-zero regions)
        all_data = []
        for fp in self.file_paths:
            image = np.load(fp)[:, :, self.start_band:]  # Shape: (height, width, num_bands - start_band)
            
            # Flatten the image and exclude zero regions
            flattened_image = image.reshape(-1, image.shape[-1])  # Shape: (height * width, num_bands - start_band)
            #non_zero_mask = np.any(flattened_image != 0, axis=1)  # Mask for non-zero pixels
            #non_zero_pixels = flattened_image[non_zero_mask]  # Shape: (num_non_zero_pixels, num_bands - start_band)
            
            all_data.append(flattened_image)

        # Concatenate all non-zero pixels along the first dimension
        all_data = np.concatenate(all_data, axis=0)  # Shape: (total_non_zero_pixels, num_bands - start_band)

        # Compute min and max over all non-zero pixels
        min_val = np.min(all_data, axis=0)  # Shape: (num_bands - start_band,)
        max_val = np.max(all_data, axis=0)  # Shape: (num_bands - start_band,)
        return min_val, max_val

def custom_collate_fn(batch):
    # Concatenate all patches into a single tensor
    patches = torch.cat(batch, dim=0)  # Shape: (total_patches, num_bands - start_band, patch_size, patch_size)
    return patches

# Hyperparameters
# num_bands = 400  # Number of spectral bands
# latent_dim = 128  # Latent dimension
# num_heads = 10  # Number of attention heads
# num_layers = 3  # Number of transformer layers
# num_selected_bands = 10  # Number of selected bands
# patch_size = 4  # Patch size
# batch_size = 2  # Batch size
# num_epochs = 30  # Number of epochs
# learning_rate = 1e-3  # Learning rate

num_bands = 400  # Number of spectral bands
latent_dim = 8  # Latent dimension
num_heads = 2  # Number of attention heads
num_layers = 2  # 3 Number of transformer layers
num_selected_bands = 10  # Number of selected bands
patch_size = 10  # Patch size
batch_size = 1  # Batch size
num_epochs = 20  # Number of epochs
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
        loss, recon, div, reg = loss_function(reconstructed, batch, indices, attention_scores, selected)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step the scheduler
    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, recon: {recon.item():.4f}, div: {div.item():.4f}, reg: {reg.item():.4f}")
    print(indices)
    #print(attention_scores)
torch.save(model.state_dict(), "new_former_10.pth")