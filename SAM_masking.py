import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from spectral import open_image, imshow, save_rgb
#from spectral.algorithms import sam
from skimage.morphology import binary_opening, binary_closing



# Function to load hyperspectral image
def load_hsi(hdr_path):
    img = open_image(hdr_path)
    return img.load()

def sam(pixel, reference):
    """ Compute Spectral Angle Mapper (SAM) similarity """
    cos_theta = np.dot(pixel, reference) / (np.linalg.norm(pixel) * np.linalg.norm(reference))
    return np.arccos(np.clip(cos_theta, -1, 1))
# Function to handle ROI selection

def onselect(eclick, erelease):
    global normal_roi, anomalous_roi
    x1, y1 = int(eclick.xdata), int(eclick.ydata)  # Top-left corner
    x2, y2 = int(erelease.xdata), int(erelease.ydata)  # Bottom-right corner

    # Ensure x1 < x2 and y1 < y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Store the ROI
    if normal_roi is None:
        normal_roi = (x1, y1, x2, y2)
        print(f"Normal ROI selected: {normal_roi}")
    else:
        anomalous_roi = (x1, y1, x2, y2)
        print(f"Anomalous ROI selected: {anomalous_roi}")
        plt.close()  # Close the plot after both ROIs are selected

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
    #sam_map = sam(hsi_flat, tomato_signature)
    sam_map = np.array([sam(px, tomato_signature) for px in hsi_flat])
    sam_map = sam_map.reshape(hsi.shape[0], hsi.shape[1])  # Reshape back to 2D

    # Create binary mask based on threshold
    mask = sam_map < threshold
    return mask

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

# Main function
def mask_tomato(hdr_path, threshold=0.1):
    global normal_roi

    # Load the hyperspectral image
    hsi = np.load(hdr_path)

    # Extract the tomato's spectral signature using the normal ROI
    #tomato_signature = extract_tomato_signature(hsi, normal_roi)
    tomato_signature = np.load('/workspace/src/Season_4/Normal/cubes/tomato_sign.npy')

    # Create the tomato mask
    mask = create_tomato_mask(hsi, tomato_signature, threshold)

    # Clean up the mask
    mask = clean_mask(mask)

    # Apply the mask to the HSI
    masked_hsi = apply_mask(hsi, mask)

    # Overlay the mask on a specific band (e.g., band 0)
    overlay = overlay_mask_on_band(hsi, mask, band_index=0)

    # Visualize the overlay
    plt.figure(figsize=(10, 10))
    plt.title("Mask Overlay on Band 0")
    plt.imshow(hsi[:, :, 140], cmap='gray')  # Display the first band
    plt.imshow(overlay, cmap='jet', alpha=0.5)  # Overlay the mask
    plt.colorbar(label="Mask Overlay")
    plt.show()

    return masked_hsi, mask
# Example usage
if __name__ == "__main__":
    # Path to the HSI file
    hdr_path = '/workspace/src/Season_4/Normal/cubes/s1_norm7 _bbox_1.npy'

    # Mask out everything except the tomato
    masked_hsi, mask = mask_tomato(hdr_path, threshold=0.25)

    # Save the masked HSI
    #np.save("masked_tomato.npy", masked_hsi)

    # Save the mask as an image (for visualization)
    #save_rgb("tomato_mask.png", mask, format='png')