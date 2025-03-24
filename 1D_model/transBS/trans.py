import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
import glob
from torch.cuda.amp import GradScaler, autocast
import math

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_dim = 448  # Number of spectral bands
latent_dim = 64  # Latent space size
num_heads = 4  
num_layers = 2  
num_selected_bands = 10  # Number of selected bands
batch_size = 400  
num_epochs = 20  
learning_rate = 0.00156 
alpha = 0.1  # Sparsity loss weight
beta = 1.0  # Diversity loss weight

# Load `.npy` files
data_dir = "/workspace/src/Season_4/Normal/sample_cubes"
file_paths = glob.glob(f"{data_dir}/*.npy")

# Resize function
def resize_cube(cube, target_height, target_width):
    height, width, num_bands = cube.shape
    zoom_factors = (target_height / height, target_width / width, 1)
    return zoom(cube, zoom_factors, order=1)

# Load and resize all cubes
cubes = [resize_cube(np.load(fp), 300, 300) for fp in file_paths]
cubes = np.stack(cubes)

# Flatten spatial dimensions
pixel_data = cubes.reshape(-1, cubes.shape[-1])

# Apply smoothing
pixel_data_smoothed = savgol_filter(pixel_data, window_length=7, polyorder=3, axis=-1)
mean = np.mean(pixel_data_smoothed, axis=0)
std = np.std(pixel_data_smoothed, axis=0)
global_mean = np.mean(pixel_data_smoothed, axis=0)
global_std = np.std(pixel_data_smoothed, axis=0)
np.save("global_mean.npy", global_mean)
np.save("global_std.npy", global_std)
pixel_data_normalized = (pixel_data_smoothed - mean) / (std + 1e-8)
pixel_data = torch.tensor(pixel_data_normalized, dtype=torch.float32).unsqueeze(1).to(device)
# Convert to PyTorch tensor (without normalization)
#pixel_data = torch.tensor(pixel_data_smoothed, dtype=torch.float32).unsqueeze(1).to(device)

# DataLoader
dataset = TensorDataset(pixel_data, pixel_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        position = torch.arange(input_dim).unsqueeze(1)  # Shape: (input_dim, 1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-math.log(10000.0) / latent_dim))
        pe = torch.zeros(1, input_dim, latent_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[0, :, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe  # Add positional encoding to the input

class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, input_dim):
        super(TransformerDecoder, self).__init__()
        # Embedding layer for the selected bands
        self.embedding = nn.Linear(latent_dim, latent_dim)
        
        # Positional encoding for the output sequence (full spectrum)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, latent_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer to map latent_dim to 1 (reconstructed spectral value)
        self.output_layer = nn.Linear(latent_dim, 1)

    def forward(self, selected_bands):
        # selected_bands shape: (batch_size, num_selected_bands, latent_dim)
        batch_size, num_selected_bands, latent_dim = selected_bands.shape
        
        # Embed the selected bands
        selected_bands = self.embedding(selected_bands)  # (batch_size, num_selected_bands, latent_dim)
        
        # Add positional encoding to the output sequence (full spectrum)
        output_seq = self.positional_encoding.expand(batch_size, -1, -1)  # (batch_size, input_dim, latent_dim)
        
        # Transformer decoder
        # selected_bands: (num_selected_bands, batch_size, latent_dim)
        # output_seq: (input_dim, batch_size, latent_dim)
        selected_bands = selected_bands.permute(1, 0, 2)  # (num_selected_bands, batch_size, latent_dim)
        output_seq = output_seq.permute(1, 0, 2)  # (input_dim, batch_size, latent_dim)
        reconstructed = self.transformer_decoder(output_seq, selected_bands)  # (input_dim, batch_size, latent_dim)
        
        # Map to output dimension
        reconstructed = self.output_layer(reconstructed.permute(1, 0, 2))  # (batch_size, input_dim, 1)
        return reconstructed.squeeze(-1)  # (batch_size, input_dim)
    
# Define Model
class BSFormerPlusPlus(nn.Module):
    def __init__(self, input_dim, latent_dim, num_heads, num_layers, num_selected_bands):
        super(BSFormerPlusPlus, self).__init__()
        self.embedding = nn.Linear(1, latent_dim)
        self.positional_encoding = SinusoidalPositionalEncoding(input_dim, latent_dim)  # Use sinusoidal encoding

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Band selection
        self.band_selection = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        nn.init.kaiming_normal_(self.band_selection[0].weight)  # Proper initialization

        # Transformer decoder for reconstruction
        self.reconstruction_decoder = TransformerDecoder(latent_dim, num_heads, num_layers, input_dim)

        self.num_selected_bands = num_selected_bands

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, num_bands, 1)
        x = self.embedding(x)  # (batch, num_bands, latent_dim)
        x = self.positional_encoding(x)  # Add sinusoidal positional encoding
        x = self.transformer_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)  # (batch, num_bands, latent_dim)

        band_weights = torch.sigmoid(self.band_selection(x)).squeeze(-1)  # (batch, num_bands)
        selected_indices = torch.topk(band_weights, self.num_selected_bands, dim=1).indices  # (batch, num_selected_bands)

        # Gather selected bands
        selected_bands = torch.gather(x, 1, selected_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # (batch, num_selected_bands, latent_dim)

        # Use transformer decoder to reconstruct full spectrum
        reconstructed = self.reconstruction_decoder(selected_bands)  # (batch, input_dim)

        return reconstructed, selected_indices, band_weights
    
    def reconstruct_from_cropped_bands(self, cropped_bands):
    
        # Embed the cropped bands
        cropped_bands = cropped_bands.permute(0, 2, 1)  # (batch, num_cropped_bands, 1)
        embedded_bands = self.embedding(cropped_bands)  # (batch, num_cropped_bands, latent_dim)

        # Add positional encoding for the output sequence (full spectrum)
        batch_size = embedded_bands.size(0)
        output_seq = self.reconstruction_decoder.positional_encoding.expand(batch_size, -1, -1)  # (batch, input_dim, latent_dim)

        # Transformer decoder
        # embedded_bands: (batch, num_cropped_bands, latent_dim)
        # output_seq: (batch, input_dim, latent_dim)
        embedded_bands = embedded_bands.permute(1, 0, 2)  # (num_cropped_bands, batch, latent_dim)
        output_seq = output_seq.permute(1, 0, 2)  # (input_dim, batch, latent_dim)
        reconstructed = self.reconstruction_decoder.transformer_decoder(output_seq, embedded_bands)  # (input_dim, batch, latent_dim)

        # Map to output dimension
        reconstructed = self.reconstruction_decoder.output_layer(reconstructed.permute(1, 0, 2))  # (batch, input_dim, 1)
        return reconstructed.squeeze(-1)  # (batch, input_dim)

# Define Loss Function
def loss_function(reconstructed, original, band_weights, selected_bands, alpha=1, beta=0, gamm=1):
    # Reconstruction loss (MSE)
    reconstruction_loss = F.mse_loss(reconstructed, original.squeeze(1))

    # Spectral angle loss (optional)
    def spectral_angle_loss(reconstructed, original):
        cos_sim = F.cosine_similarity(reconstructed, original, dim=1)
        return torch.mean(1 - cos_sim)
    spectral_loss = spectral_angle_loss(reconstructed, original.squeeze(1))

    # Sparsity loss (L1 regularization)
    sparsity_loss = torch.mean(torch.abs(band_weights))

    # Diversity loss (optimized)
    def diversity_loss(selected_bands, min_distance=3):
        selected_bands = selected_bands.unsqueeze(2)  # (batch, num_selected, 1)
        pairwise_distances = torch.abs(selected_bands - selected_bands.transpose(1, 2))  # (batch, num_selected, num_selected)
        mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1)  # Upper triangular mask
        loss = torch.mean(torch.exp(-pairwise_distances / min_distance) * mask)
        return loss
    diversity_penalty = diversity_loss(selected_bands)

    # Total loss
    total_loss = reconstruction_loss + gamm * spectral_loss + alpha * sparsity_loss + beta * diversity_penalty
    return total_loss, reconstruction_loss, spectral_loss, sparsity_loss, diversity_penalty

# Initialize Model
model = BSFormerPlusPlus(input_dim, latent_dim, num_heads, num_layers, num_selected_bands).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = GradScaler()  # Mixed precision training

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        with autocast():
            reconstructed, selected_bands, band_weights = model(data)
            loss, reconstruction_loss, spectral_loss, sparsity_loss, diversity_penalty = loss_function(reconstructed, target, band_weights, selected_bands, alpha, beta)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Total_Loss: {loss.item():.8f}, recon_loss: {reconstruction_loss.item():.8f}, spectral_loss {spectral_loss.item():.8f}, sparsity_loss: {sparsity_loss.item():.8f}, diversity_penalty: {diversity_penalty.item():.8f}")

# Save Model
torch.save(model.state_dict(), "bsformer_plus_plus.pth")
torch.save({
    'decoder_state_dict': model.reconstruction_decoder.state_dict(),
    'embedding_state_dict': model.embedding.state_dict(),
}, 'decoder_and_embedding.pth')