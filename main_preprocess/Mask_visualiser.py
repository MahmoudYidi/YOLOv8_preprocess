import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Global variables
current_index = 0
file_list = []
band_index = 120  # Band to visualize

# Function to display the current image
def display_image():
    global current_index, file_list, band_index, fig, canvas

    if not file_list or current_index >= len(file_list):
        print("No more files to process.")
        return

    # Load the current .npy file
    file_path = file_list[current_index]
    hsi = np.load(file_path)

    # Validate band index
    if band_index >= hsi.shape[2]:
        print(f"Band index {band_index} is out of range.")
        return

    band = hsi[:, :, band_index]

    # Clear the plot without destroying the frame
    ax.clear()
    ax.set_title(f"Band {band_index} - {os.path.basename(file_path)}")
    ax.imshow(band, cmap='gray')  # Display image without colorbar

    # Redraw canvas
    canvas.draw()

# Function to load next image
def next_image():
    global current_index
    if current_index < len(file_list) - 1:
        current_index += 1
        display_image()
    else:
        print("No more images.")

# Function to select a directory
def select_directory():
    global file_list, current_index
    directory = filedialog.askdirectory()
    if directory:
        file_list = sorted(
            [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
        )
        if file_list:
            current_index = 0
            display_image()
        else:
            print("No .npy files found.")

# Create main window
root = tk.Tk()
root.title("HSI Band Visualizer")

# Create a matplotlib figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Create a button frame
button_frame = tk.Frame(root)
button_frame.pack(side=tk.BOTTOM, fill=tk.X)

# Buttons
select_button = tk.Button(button_frame, text="Select Directory", command=select_directory)
select_button.pack(side=tk.LEFT, padx=10, pady=10)

next_button = tk.Button(button_frame, text="Next", command=next_image)
next_button.pack(side=tk.LEFT, padx=10, pady=10)

# Start Tkinter loop
root.mainloop()
