import os
import json
import numpy as np
from spectral import open_image

# Function to load hyperspectral image
def load_hsi(hdr_path):
    img = open_image(hdr_path)
    return img

# Function to normalize hyperspectral image
def normalize_hsi(original, dark, white):
    # Normalize using the formula: (original - dark) / (white - dark + 1e-6)
    calibrated_hsi = (original - dark) / (white - dark + 1e-6)
    # Handle NaN and infinite values
    calibrated_hsi = np.nan_to_num(calibrated_hsi, nan=0.0, posinf=0.0, neginf=0.0)
    return calibrated_hsi

# Function to crop hyperspectral image using bounding box
def crop_hsi(img, bbox):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2, :]

# Function to save cropped hyperspectral cube
def save_cropped_hsi(cropped_img, output_path):
    np.save(output_path, cropped_img)

# Main function
def process_hsi_with_bboxes(json_path, hsi_root_dir, output_dir, dark_path, white_path):
    # Load JSON file
    with open(json_path, "r") as f:
        bbox_data = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dark and white reference images
    dark_img = load_hsi(dark_path).load()
    white_img = load_hsi(white_path).load()

    # Iterate through each entry in the JSON
    for key, bboxes in bbox_data.items():
        # Construct the path to the corresponding HSI file
        hsi_folder = os.path.join(hsi_root_dir, key, "capture")
        hdr_file = os.path.join(hsi_folder, f"{key}.hdr")

        if not os.path.exists(hdr_file):
            print(f"HSI file not found for {key}: {hdr_file}")
            continue

        # Load the hyperspectral image
        hsi_img = load_hsi(hdr_file).load()

        # Normalize the hyperspectral image
        calibrated_hsi = normalize_hsi(hsi_img, dark_img, white_img)

        # Iterate through each bounding box for the current image
        for i, bbox in enumerate(bboxes):
            # Crop the normalized hyperspectral image
            cropped_cube = crop_hsi(calibrated_hsi, bbox)

            # Save the cropped cube
            output_filename = f"{key}_bbox_{i+1}.npy"
            output_path = os.path.join(output_dir, output_filename)
            save_cropped_hsi(cropped_cube, output_path)
            print(f"Saved cropped cube: {output_path}")

# Example usage
if __name__ == "__main__":
    # Path to the JSON file with bounding box coordinates
    #json_path = "/workspace/src/Season_4/normal.json"
    json_path = "/workspace/src/test/testing.json"

    # Root directory containing HSI folders (e.g., s1_norm1, s1_norm2, etc.)
    #hsi_root_dir = "/workspace/src/Season_4/Normal/hsi"
    hsi_root_dir = "/workspace/src/test"

    # Output directory to save cropped hyperspectral cubes
    #output_dir = "/workspace/src/Season_4/Normal/cubes"
    output_dir = "/workspace/src/test/test_cubes"

    # Paths to dark and white reference images
    #dark_path = "/workspace/src/Season_4/dark_ref/capture/dark_ref.hdr"
    #white_path = "/workspace/src/Season_4/white_ref/capture/white_ref.hdr"
    dark_path = "/workspace/src/dataset/hsi/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr"
    white_path = "/workspace/src/dataset/hsi/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr"

    # Process the HSI images using the bounding boxes
    process_hsi_with_bboxes(json_path, hsi_root_dir, output_dir, dark_path, white_path)