import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from scipy.signal import savgol_filter
import math

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sinusoidal Positional Encoding
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SinusoidalPositionalEncoding, self).__init__()
        position = torch.arange(input_dim).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-math.log(10000.0) / latent_dim))
        pe = torch.zeros(1, input_dim, latent_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim, num_heads, num_layers, input_dim):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Linear(latent_dim, latent_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, latent_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=256, dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(latent_dim, 1)

    def forward(self, selected_bands):
        batch_size, num_selected_bands, latent_dim = selected_bands.shape
        selected_bands = self.embedding(selected_bands)
        output_seq = self.positional_encoding.expand(batch_size, -1, -1)
        selected_bands = selected_bands.permute(1, 0, 2)
        output_seq = output_seq.permute(1, 0, 2)
        reconstructed = self.transformer_decoder(output_seq, selected_bands)
        reconstructed = self.output_layer(reconstructed.permute(1, 0, 2))
        return reconstructed.squeeze(-1)

# Load the decoder and embedding
input_dim = 398
latent_dim = 64
num_heads = 4
num_layers = 2
decoder_checkpoint_path = "decoder_and_embedding.pth"
checkpoint = torch.load(decoder_checkpoint_path, map_location=device)

embedding = nn.Linear(1, latent_dim).to(device)
decoder = TransformerDecoder(latent_dim, num_heads, num_layers, input_dim).to(device)
embedding.load_state_dict(checkpoint['embedding_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
embedding.eval()
decoder.eval()

# Load test data
test_cube_path = "/workspace/src/test/test_cubes/test_bbox_3.npy"
test_cube = np.load(test_cube_path)
test_cube = test_cube[:,:, 50:]

global_mean = np.load("global_mean.npy")
global_std = np.load("global_std.npy")

fixed_bands = [ 19,  20,  18,   1,   2,   0, 320, 243, 241, 244]

def reconstruct_with_fixed_bands_cropped(spectral_signature, fixed_bands):
    smoothed_signature = savgol_filter(spectral_signature, window_length=7, polyorder=3)
    normalized_signature = (smoothed_signature - global_mean) / (global_std + 1e-8)
    cropped_signature = normalized_signature[fixed_bands]
    cropped_signature_tensor = torch.tensor(cropped_signature, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    embedded_bands = embedding(cropped_signature_tensor)

    batch_size = embedded_bands.size(0)
    output_seq = decoder.positional_encoding.expand(batch_size, -1, -1)
    embedded_bands = embedded_bands.permute(1, 0, 2)
    output_seq = output_seq.permute(1, 0, 2)

    reconstructed = decoder.transformer_decoder(output_seq, embedded_bands)
    reconstructed = decoder.output_layer(reconstructed.permute(1, 0, 2)).squeeze().cpu().detach().numpy()
    reconstructed_spectrum = reconstructed * global_std + global_mean

    return smoothed_signature, reconstructed_spectrum

def compute_reconstruction_errors(test_cube, fixed_bands):
    height, width, num_bands = test_cube.shape
    reconstruction_errors = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            spectral_signature = test_cube[y, x, :]
            smoothed, reconstructed = reconstruct_with_fixed_bands_cropped(spectral_signature, fixed_bands)
            reconstruction_errors[y, x] = np.mean((smoothed - reconstructed) ** 2)

    return reconstruction_errors

reconstruction_errors = compute_reconstruction_errors(test_cube, fixed_bands)

mean_image = np.mean(test_cube, axis=2)
mean_image_normalized = (mean_image - np.min(mean_image)) / (np.max(mean_image) - np.min(mean_image))
mean_image_normalized = (mean_image_normalized * 255).astype(np.uint8)
mean_image_pil = Image.fromarray(mean_image_normalized)

# GUI Application
class ReconstructionErrorApp:
    def __init__(self, root, mean_image_pil, reconstruction_errors):
        self.root = root
        self.mean_image_pil = mean_image_pil
        self.reconstruction_errors = reconstruction_errors
        self.threshold = 0.0

        self.canvas = tk.Canvas(root, width=mean_image_pil.width, height=mean_image_pil.height)
        self.canvas.pack()

        self.threshold_slider = ttk.Scale(root, from_=0, to=np.max(reconstruction_errors)*0.5, orient="horizontal", command=self.update_overlay)
        self.threshold_slider.pack(fill="x", padx=20, pady=10)

        self.threshold_label = ttk.Label(root, text=f"Threshold: {self.threshold:.4f}")
        self.threshold_label.pack()

        self.tk_image = ImageTk.PhotoImage(self.mean_image_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def update_overlay(self, value):
        self.threshold = float(value)
        overlay_image = self.mean_image_pil.copy()
        overlay = np.zeros_like(self.reconstruction_errors)
        overlay[self.reconstruction_errors > self.threshold] = 255

        red_overlay = Image.new("RGBA", overlay_image.size, (255, 0, 0, 0))
        red_overlay.putalpha(Image.fromarray(overlay.astype(np.uint8)))

        combined_image = Image.alpha_composite(overlay_image.convert("RGBA"), red_overlay)
        self.tk_image = ImageTk.PhotoImage(combined_image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.threshold_label.config(text=f"Threshold: {self.threshold:.4f}")

root = tk.Tk()
root.title("Reconstruction Error Threshold")
app = ReconstructionErrorApp(root, mean_image_pil, reconstruction_errors)
root.mainloop()
