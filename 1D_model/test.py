import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from scipy.ndimage import zoom  # For resizing
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.signal import savgol_filter

# Load the trained model
class Autoencoder1D(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder1D, self).__init__()
        # Define the encoder and decoder (same as before)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (input_dim // 4), latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (input_dim // 4)),
            nn.Unflatten(1, (32, input_dim // 4)),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Hyperparameters
input_dim = 448  # Number of spectral bands
latent_dim = 10  # Latent space dimension
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_save_path = "/workspace/src/Season_4/Normal/trained_autoencoder.pth"
model = Autoencoder1D(input_dim, latent_dim).to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

# Load the hyperspectral cube for testing
#test_cube_path = "/workspace/src/test/test_masked/test_bbox_3.npy"  # Replace with your test cube path
test_cube_path = "/workspace/src/test/test_cubes/test_bbox_3.npy"
test_cube = np.load(test_cube_path)  # Shape: (height, width, num_bands)

# Compute the mean across spectral bands for visualization
mean_image = np.mean(test_cube, axis=2)  # Shape: (height, width)

# Function to handle mouse clicks
def onclick(event):
    if event.inaxes is not None:
        # Get the clicked pixel coordinates
        x, y = int(event.xdata), int(event.ydata)
        print(f"Clicked pixel coordinates: ({x}, {y})")

        # Extract the spectral signature of the clicked pixel
        spectral_signature = test_cube[y, x, :]  # Shape: (num_bands,)
        #spectral_signature_tensor = torch.tensor(spectral_signature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Shape: (1, 1, num_bands)
        
        
        #############################################
        # Apply Savitzky-Golay filter to smooth the spectral signature
        smoothed_signature = savgol_filter(spectral_signature, window_length=7, polyorder=3)

        # Convert to PyTorch tensor, add channel dimension, and move to GPU
        spectral_signature_tensor = torch.tensor(smoothed_signature, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        #############################################################################################

        # Reconstruct the spectral signature using the trained model
        with torch.no_grad():
            reconstructed_signature = model(spectral_signature_tensor).squeeze().cpu().numpy()  # Shape: (num_bands,)

        # Compute the reconstruction error
        reconstruction_error = np.mean((spectral_signature - reconstructed_signature) ** 2)
        print(f"Reconstruction error: {reconstruction_error:.4f}")

        # Plot the original and reconstructed spectral signatures
        plt.figure()
        #plt.plot(spectral_signature, label="Original")
        plt.plot(smoothed_signature , label="Original")
        plt.plot(reconstructed_signature, label="Reconstructed")
        plt.title(f"Spectral Signature at ({x}, {y})")
        plt.xlabel("Spectral Band")
        plt.ylabel("Intensity")
        plt.legend()
        plt.show()

# Plot the mean image and enable mouse clicks
fig, ax = plt.subplots()
ax.imshow(mean_image, cmap="gray")
ax.set_title("Click on a pixel to test")
fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()