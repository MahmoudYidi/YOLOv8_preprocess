import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Global variables
current_index = 0
file_list = []
band_index = 120  # Band to visualize (e.g., band 0)

# Function to load and display the current image
def display_image():
    global current_index, file_list, band_index

    if current_index >= len(file_list):
        print("No more files to process.")
        return

    # Load the current .npy file
    file_path = file_list[current_index]
    hsi = np.load(file_path)

    # Check if the band index is valid
    if band_index >= hsi.shape[2]:
        print(f"Band index {band_index} is out of range for this image.")
        return

    # Extract the specific band
    band = hsi[:, :, band_index]

    # Clear the previous plot
    ax.clear()

    # Display the band
    ax.set_title(f"Band {band_index} of {os.path.basename(file_path)}")
    im = ax.imshow(band, cmap='gray')

    # Add a colorbar
    if hasattr(display_image, 'colorbar'):
        display_image.colorbar.remove()  # Remove the previous colorbar
    display_image.colorbar = plt.colorbar(im, ax=ax, label="Intensity")

    # Draw the plot on the Tkinter canvas
    canvas.draw()

# Function to handle the "Next" button click
def next_image():
    global current_index
    current_index += 1
    if current_index < len(file_list):
        display_image()
    else:
        print("Reached the end of the file list.")

# Function to select a directory and load .npy files
def select_directory():
    global file_list, current_index
    directory = filedialog.askdirectory()
    if directory:
        file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        file_list.sort()  # Sort files alphabetically
        if file_list:
            current_index = 0
            display_image()
        else:
            print("No .npy files found in the selected directory.")

# Create the main GUI window
root = tk.Tk()
root.title("HSI Band Visualizer")

# Create a frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Add a "Select Directory" button
select_button = tk.Button(button_frame, text="Select Directory", command=select_directory)
select_button.pack(side=tk.LEFT, padx=10, pady=10)

# Add a "Next" button
next_button = tk.Button(button_frame, text="Next", command=next_image)
next_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create a matplotlib figure and embed it in the Tkinter window
fig, ax = plt.subplots(figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Run the Tkinter event loop
root.mainloop()