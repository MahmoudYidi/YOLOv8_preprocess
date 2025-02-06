import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from skimage.morphology import binary_opening, binary_closing

# Global variables
current_index = 0
file_list = []
tomato_signature = None

# Function to compute SAM
def sam(pixel, reference):
    """Compute Spectral Angle Mapper (SAM) similarity."""
    cos_theta = np.dot(pixel, reference) / (np.linalg.norm(pixel) * np.linalg.norm(reference))
    return np.arccos(np.clip(cos_theta, -1, 1))

# Function to create a mask for the tomato
def create_tomato_mask(hsi, tomato_signature, threshold=0.17):
    hsi_flat = hsi.reshape(-1, hsi.shape[2])  # Flatten spatial dimensions
    sam_map = np.array([sam(px, tomato_signature) for px in hsi_flat])
    sam_map = sam_map.reshape(hsi.shape[0], hsi.shape[1])  # Reshape back to 2D
    mask = sam_map < threshold
    return mask

# Function to clean up the mask
def clean_mask(mask):
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

# Function to process and visualize the current image
def process_and_visualize():
    global current_index, file_list, tomato_signature

    if current_index >= len(file_list):
        print("No more files to process.")
        return

    # Load the current HSI file
    hsi_path = file_list[current_index]
    hsi = np.load(hsi_path)

    # Create the tomato mask
    mask = create_tomato_mask(hsi, tomato_signature, threshold=0.17)
    mask = clean_mask(mask)

    # Apply the mask to the HSI
    masked_hsi = apply_mask(hsi, mask)

    # Overlay the mask on a specific band (e.g., band 0)
    overlay = overlay_mask_on_band(hsi, mask, band_index=0)

    # Convert the overlay to a PIL image
    overlay_normalized = (overlay - overlay.min()) / (overlay.max() - overlay.min())  # Normalize to [0, 1]
    overlay_normalized = (overlay_normalized * 255).astype(np.uint8)  # Scale to [0, 255]
    overlay_image = Image.fromarray(overlay_normalized)

    # Resize the image to fit the GUI window
    overlay_image = overlay_image.resize((400, 400), Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS

    # Convert the PIL image to a Tkinter-compatible image
    tk_image = ImageTk.PhotoImage(overlay_image)

    # Update the image in the GUI
    image_label.config(image=tk_image)
    image_label.image = tk_image  # Keep a reference to avoid garbage collection

# Function to handle the "Next" button click
def next_image():
    global current_index
    current_index += 1
    if current_index < len(file_list):
        process_and_visualize()
    else:
        print("Reached the end of the file list.")

# Function to select a directory and load files
def select_directory():
    global file_list, tomato_signature, current_index
    directory = filedialog.askdirectory()
    if directory:
        file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        file_list.sort()  # Sort files alphabetically
        if file_list:
            current_index = 0
            tomato_signature = np.load('/workspace/src/Season_4/Normal/cubes/tomato_sign.npy')
            process_and_visualize()
        else:
            print("No .npy files found in the selected directory.")

# Create the main GUI window
root = tk.Tk()
root.title("Tomato Masking GUI")

# Create a frame for the image
image_frame = tk.Frame(root)
image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create a label to display the image
image_label = tk.Label(image_frame)
image_label.pack(fill=tk.BOTH, expand=True)

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Add a "Select Directory" button
select_button = tk.Button(button_frame, text="Select Directory", command=select_directory)
select_button.pack(side=tk.LEFT, padx=10, pady=10)

# Add a "Next" button
next_button = tk.Button(button_frame, text="Next", command=next_image)
next_button.pack(side=tk.LEFT, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()