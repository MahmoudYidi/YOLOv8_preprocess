import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.morphology import binary_opening, binary_closing
import glob
# Global variables to store ROIs
normal_roi = None

# Function to compute SAM
def sam(pixel, reference):
    """Compute Spectral Angle Mapper (SAM) similarity."""
    cos_theta = np.dot(pixel, reference) / (np.linalg.norm(pixel) * np.linalg.norm(reference))
    return np.arccos(np.clip(cos_theta, -1, 1))

# Function to handle ROI selection
def onselect(eclick, erelease):
    global normal_roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Top-left corner
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Bottom-right corner

    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Store the ROI
    normal_roi = (x1, y1, x2, y2)
    print(f"Normal ROI selected: {normal_roi}")
    plt.close()  # Close the plot after ROI is selected

# Function to extract the tomato's spectral signature
def extract_tomato_signature(hsi, roi):
    x1, y1, x2, y2 = roi
    tomato_pixels = hsi[y1:y2, x1:x2, :]
    tomato_signature = np.mean(tomato_pixels, axis=(0, 1))  # Average spectrum
    return tomato_signature

# Function to create a mask for the tomato
def create_tomato_mask(hsi, tomato_signature, threshold=0.1):
    # Compute Spectral Angle Mapper (SAM) similarity
    hsi_flat = hsi.reshape(-1, hsi.shape[2])  # Flatten spatial dimensions
    sam_map = np.array([sam(px, tomato_signature) for px in hsi_flat])
    sam_map = sam_map.reshape(hsi.shape[0], hsi.shape[1])  # Reshape back to 2D

    # Create binary mask based on threshold
    mask = sam_map < threshold
    return mask, sam_map

# Function to clean up the mask
def clean_mask(mask):
    # Remove small objects and smooth the mask
    mask = binary_opening(mask)  # Remove small noise
    mask = binary_closing(mask)  # Fill small holes
    return mask

# Function to apply the mask to the HSI
def apply_mask(hsi, mask):
    masked_hsi = np.zeros_like(hsi)
    masked_hsi[mask] = hsi[mask]
    return masked_hsi

# Function to overlay the mask on a specific band
def overlay_mask_on_band(hsi, mask, band_index):
    band = hsi[:, :, band_index]
    overlay = np.ma.masked_where(~mask, band)  # Mask everything except the tomato
    return overlay

# Function to select ROI and extract reference signature
def select_roi_and_extract_signature(hdr_path):
    global normal_roi

    # Load the hyperspectral image
    hsi = np.load(hdr_path)

    # Display the first band for ROI selection
    plt.figure(figsize=(10, 10))
    plt.title("Select the ROI for the tomato (Normal)")
    plt.imshow(hsi[:, :, 146], cmap='gray')  # Display a specific band

    # Create a RectangleSelector widget
    rs = RectangleSelector(
        plt.gca(),
        onselect,
        useblit=True,
        button=[1],
        minspanx=2,
        minspany=2,
        spancoords='pixels',
        interactive=True
    )

    plt.show()

    # Ensure the ROI is selected
    if normal_roi is None:
        raise ValueError("ROI must be selected.")

    # Extract the tomato's spectral signature using the normal ROI
    tomato_signature = extract_tomato_signature(hsi, normal_roi)
    np.save('/workspace/src/Season_4/Normal/cubes/tomato_sign.npy', tomato_signature)
    print("Tomato signature saved.")

    return tomato_signature

# Function to segment tomatoes in other images
def segment_tomato_in_image(hdr_path, tomato_signature, threshold=0.05):
    # Load the hyperspectral image
    hsi = np.load(hdr_path)

    # Create the tomato mask
    mask, sam_map = create_tomato_mask(hsi, tomato_signature, threshold)

    # Clean up the mask
    mask = clean_mask(mask)

    # Apply the mask to the HSI
    masked_hsi = apply_mask(hsi, mask)

    # Overlay the mask on a specific band (e.g., band 0)
    overlay = overlay_mask_on_band(hsi, mask, band_index=0)

    # Visualize the overlay
    plt.figure(figsize=(10, 10))
    plt.title(f"Mask Overlay on Band 0: {hdr_path}")
    plt.imshow(hsi[:, :, 140], cmap='gray')  # Display a specific band
    plt.imshow(overlay, cmap='jet', alpha=0.5)  # Overlay the mask
    plt.colorbar(label="Mask Overlay")
    plt.show()

    return masked_hsi, mask

if __name__ == "__main__":
    # Step 1: Select ROI and extract reference signature from one image
    reference_image_path = "/workspace/src/Season_4/Normal/cubes/s1_norm10_bbox_2.npy"
    tomato_signature = select_roi_and_extract_signature(reference_image_path)

    # Step 2: Read all .npy files in the directory dynamically
    directory_path = "/workspace/src/Season_4/Normal/cubes/"
    other_image_paths = glob.glob(f"{directory_path}/*.npy")

    # Exclude the reference image and signature file from processing
    other_image_paths = [
        path for path in other_image_paths 
        if path != reference_image_path and not path.endswith("tomato_sign.npy")
    ]

    # Process each image
    for image_path in other_image_paths:
        print(f"Processing {image_path}...")
        masked_hsi, mask = segment_tomato_in_image(image_path, tomato_signature, threshold=0.17)