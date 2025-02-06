import os
import numpy as np
from skimage.morphology import binary_opening, binary_closing
import glob

# Function to compute SAM
def sam(pixel, reference):
    """Compute Spectral Angle Mapper (SAM) similarity."""
    cos_theta = np.dot(pixel, reference) / (np.linalg.norm(pixel) * np.linalg.norm(reference))
    return np.arccos(np.clip(cos_theta, -1, 1))

# Function to create a mask for the tomato
def create_tomato_mask(hsi, tomato_signature, threshold=0.1):
    """Create a binary mask for the tomato using SAM."""
    hsi_flat = hsi.reshape(-1, hsi.shape[2])  # Flatten spatial dimensions
    sam_map = np.array([sam(px, tomato_signature) for px in hsi_flat])
    sam_map = sam_map.reshape(hsi.shape[0], hsi.shape[1])  # Reshape back to 2D
    mask = sam_map < threshold
    return mask

# Function to clean up the mask
def clean_mask(mask):
    """Remove noise and fill holes in the mask."""
    mask = binary_opening(mask)  # Remove small noise
    mask = binary_closing(mask)  # Fill small holes
    return mask

# Function to apply the mask to the HSI
def apply_mask(hsi, mask):
    """Set background pixels to 0 using the mask."""
    masked_hsi = np.zeros_like(hsi)
    masked_hsi[mask] = hsi[mask]  # Keep foreground (tomato) pixels
    return masked_hsi

# Function to process all images in a directory
def process_images_in_directory(input_dir, output_dir, tomato_signature, threshold=0.17):
    """Process all .npy files in the input directory and save masked images to the output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all .npy files in the input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.npy"))

    # Process each image
    for image_path in image_paths:
        print(f"Processing {os.path.basename(image_path)}...")

        # Load the hyperspectral image
        hsi = np.load(image_path)

        # Create the tomato mask
        mask = create_tomato_mask(hsi, tomato_signature, threshold)
        mask = clean_mask(mask)

        # Apply the mask to the HSI
        masked_hsi = apply_mask(hsi, mask)

        # Save the masked image to the output directory
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        np.save(output_path, masked_hsi)

        print(f"Saved masked image to {output_path}")

# Main script
if __name__ == "__main__":
    # Paths
    input_directory = "/workspace/src/Season_4/Normal/cubes/"  # Directory containing .npy files
    output_directory = "/workspace/src/Season_4/Normal/masked_cubes/"  # Directory to save masked images
    tomato_signature_path = "/workspace/src/Season_4/Normal/tomato_sign.npy"  # Path to the tomato signature

    # Load the tomato signature
    tomato_signature = np.load(tomato_signature_path)
    print("Tomato signature loaded.")

    # Process all images in the input directory
    process_images_in_directory(input_directory, output_directory, tomato_signature, threshold=0.17)