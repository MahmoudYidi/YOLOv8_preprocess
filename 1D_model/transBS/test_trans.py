import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import glob
from scipy.ndimage import zoom
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import math

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the BSFormer++ Model with Transformer Decoder
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

# Hyperparameters
input_dim = 398  # Number of spectral bands
latent_dim = 64  # Latent dimension for the Transformer
num_heads = 4  # Number of attention heads
num_layers = 2  # Number of Transformer encoder layers
num_selected_bands = 10  # Number of selected bands

# Load the trained BSFormer++ model
model_save_path = "/workspace/bsformer_plus_plus.pth"  # Replace with your model path
model = BSFormerPlusPlus(input_dim=input_dim, latent_dim=latent_dim, num_heads=num_heads, num_layers=num_layers, num_selected_bands=num_selected_bands).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

# Load the hyperspectral cube for testing
#test_cube_path = "/workspace/src/test/test_cubes/test_bbox_3.npy"  # Replace with your test cube path
test_cube_path = "/workspace/src/Season_4/Normal/sample_cubes/s1_norm13_bbox_2.npy"
test_cube = np.load(test_cube_path)  # Shape: (height, width, num_bands)
test_cube = test_cube[:,:, 50:]
# Compute the mean across spectral bands for visualization
mean_image = np.mean(test_cube, axis=2)  # Shape: (height, width)
global_mean = np.load("global_mean.npy")
global_std = np.load("global_std.npy")
# Function to handle mouse clicks
def onclick(event):
    if event.inaxes is not None:
        # Get the clicked pixel coordinates
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked pixel coordinates: ({x}, {y})")

        # Extract the spectral signature of the clicked pixel
        spectral_signature = test_cube[y, x, :]  # Shape: (num_bands,)

        # Apply Savitzky-Golay filter to smooth the spectral signature
        #smoothed_signature = savgol_filter(spectral_signature, window_length=7, polyorder=3)
        smoothed_signature = spectral_signature
        # Normalize the spectral signature (using the same mean and std as during training)
        normalized_signature = (smoothed_signature - global_mean) / (global_std + 1e-8)

        # Convert to PyTorch tensor, add channel dimension, and move to GPU
        spectral_signature_tensor = torch.tensor(normalized_signature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, num_bands)

        # Reconstruct the spectral signature using the trained BSFormer++ model
        with torch.no_grad():
            reconstructed_signature, selected_bands, band_weights = model(spectral_signature_tensor)
            reconstructed_signature = reconstructed_signature.squeeze().cpu().numpy()  # Shape: (num_bands,)
            selected_bands = selected_bands.squeeze().cpu().numpy()  # Shape: (num_selected_bands,)
            band_weights = band_weights.squeeze().cpu().numpy()  # Shape: (num_bands,)

        # Denormalize the reconstructed signature
        reconstructed_signature = reconstructed_signature *  global_std + global_mean

        # Print the selected bands
        print(f"Selected bands: {selected_bands}")

        # Compute the reconstruction error
        reconstruction_error = np.mean((smoothed_signature - reconstructed_signature) ** 2)
        print(f"Reconstruction error: {reconstruction_error:.4f}")

        # Plot the original and reconstructed spectral signatures
        plt.figure(figsize=(12, 6))

        # Plot 1: Original vs Reconstructed Spectral Signature
        plt.subplot(1, 2, 1)
        plt.plot(smoothed_signature, label="Original (Smoothed)")
        plt.plot(reconstructed_signature, label="Reconstructed")
        plt.title(f"Spectral Signature at ({x}, {y})")
        plt.xlabel("Spectral Band")
        plt.ylabel("Intensity")
        plt.legend()

        # Plot 2: Band Weights (Importance of Each Band)
        plt.subplot(1, 2, 2)
        plt.bar(range(input_dim), band_weights, color="blue", alpha=0.7)
        plt.scatter(selected_bands, band_weights[selected_bands], color="red", label="Selected Bands")
        plt.title("Band Weights (Attention)")
        plt.xlabel("Spectral Band")
        plt.ylabel("Attention Weight")
        plt.legend()

        plt.tight_layout()
        plt.show()

# Plot the mean image and enable mouse clicks
fig, ax = plt.subplots()
ax.imshow(mean_image, cmap="gray")
ax.set_title("Click on a pixel to test")
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()