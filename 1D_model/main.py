import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from scipy.ndimage import zoom  # For resizing
from scipy.signal import savgol_filter


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the 1D-CNN Autoencoder (same as before)
class Autoencoder1D(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder1D, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(0.3) , ########DropOut
            nn.Linear(32 * (input_dim // 4), latent_dim),
            nn.ReLU()
        )
        
        # Decoder
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
latent_dim = 5  # Latent space dimension
batch_size = 16
learning_rate = 0.001
num_epochs = 20

# Create the model and move it to the GPU
model = Autoencoder1D(input_dim, latent_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load all `.npy` files in a directory
data_dir = "/workspace/src/Season_4/Normal/sample_cubes"  # Replace with the path to your `.npy` files
file_paths = glob.glob(f"{data_dir}/*.npy")  # List all `.npy` files

# Define target spatial size
target_height, target_width = 300, 300  # Set your desired spatial size

# Function to resize a cube
def resize_cube(cube, target_height, target_width):
    """
    Resize a hyperspectral cube to the target spatial dimensions using interpolation.
    Args:
        cube: Input cube of shape (height, width, num_bands).
        target_height: Target height.
        target_width: Target width.
    Returns:
        Resized cube of shape (target_height, target_width, num_bands).
    """
    height, width, num_bands = cube.shape
    zoom_factors = (target_height / height, target_width / width, 1)  # Keep spectral dimension unchanged
    resized_cube = zoom(cube, zoom_factors, order=1)  # Use order=1 for bilinear interpolation
    return resized_cube

# Load and resize all cubes
cubes = []
for file_path in file_paths:
    cube = np.load(file_path)  # Load the `.npy` file
    resized_cube = resize_cube(cube, target_height, target_width)  # Resize the cube
    cubes.append(resized_cube)

# Stack cubes into a 4D tensor: (num_cubes, target_height, target_width, num_bands)
cubes = np.stack(cubes)  # Shape: (num_cubes, target_height, target_width, num_bands)

# Reshape the cubes into 2D: (num_cubes * target_height * target_width, num_bands)
pixel_data = cubes.reshape(-1, cubes.shape[-1])  # Shape: (num_cubes * target_height * target_width, num_bands)

# Apply Savitzky-Golay filter to smooth the spectral bands
window_length = 7  
polyorder = 3 

# Apply filtering along the spectral dimension (last axis)
pixel_data_smoothed = savgol_filter(pixel_data, window_length=window_length, polyorder=polyorder, axis=-1)

# Convert to PyTorch tensor, add channel dimension, and move to GPU
pixel_data = torch.tensor(pixel_data_smoothed, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (num_cubes * target_height * target_width, 1, num_bands)

# Convert to PyTorch tensor, add channel dimension, and move to GPU
#pixel_data = torch.tensor(pixel_data, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (num_cubes * target_height * target_width, 1, num_bands)

# Create a DataLoader for mini-batch training
dataset = TensorDataset(pixel_data, pixel_data)  # Input and target are the same
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch in dataloader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed = model(inputs)
        
        # Compute loss
        loss = criterion(reconstructed, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.6f}")

# Save the trained model
model_save_path = "/workspace/src/Season_4/Normal/trained_autoencoder1.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

