import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import exposure, filters



def display_band(hsi, band):
    plt.imshow(hsi[:, :, band], cmap='gray')
    plt.title('First Band of Calibrated HSI')
    plt.colorbar(label='Reflectance')
    plt.show()




hsi_data = np.load('/workspace/src/Season_4/Normal/cubes/s1_norm5_bbox_3.npy')

# Normalize the calibrated hyperspectral data
hsi_normalized = hsi_data / np.max(hsi_data)

#display_band(hsi_normalized, 30)

# Display one of the bands (e.g., the first band)
band_to_display = 152  # Change this to display a different band
plt.figure(figsize=(8, 8))
plt.imshow(hsi_normalized[:, :, band_to_display], cmap='gray')
plt.title('Select ROIs: Draw a box for Normal and Anomalous Regions')
plt.colorbar(label='Reflectance')

# Initialize lists to store ROIs
normal_roi = None
anomalous_roi = None

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

# Create a RectangleSelector widget
rs = RectangleSelector(
    plt.gca(),
    onselect,
    useblit=True,
    button=[1],  
    minspanx=5,
    minspany=5,
    spancoords='pixels',
    interactive=True
)

plt.show()

# Extract the ROIs from the hyperspectral data
if normal_roi and anomalous_roi:
    x1_n, y1_n, x2_n, y2_n = normal_roi
    x1_a, y1_a, x2_a, y2_a = anomalous_roi

    # Extract the normal and anomalous regions
    normal_region = hsi_data[y1_n:y2_n, x1_n:x2_n, :]
    anomalous_region = hsi_data[y1_a:y2_a, x1_a:x2_a, :]

    print("Normal region shape:", normal_region.shape)
    print("Anomalous region shape:", anomalous_region.shape)
else:
    print("ROI selection was not completed.")

mean_spectrum_split = np.mean(anomalous_region, axis=(0, 1))
mean_spectrum_normal = np.mean(normal_region, axis=(0, 1))

plt.plot(mean_spectrum_split, label='Split Region')
plt.plot(mean_spectrum_normal, label='Normal Region')
plt.xlabel('Band Number')
plt.ylabel('Reflectance')
plt.title('Mean Spectrum of Split vs Normal Regions')
plt.legend()
plt.show()

# Statistically. 
band_differences = np.abs(mean_spectrum_split - mean_spectrum_normal)
N = 2 # Number of bands to select
selected_bands = np.argsort(band_differences)[-N:]
#selected_bands = np.argsort(band_differences)[:N]

print(f"Selected Bands: {selected_bands}")
selected_data = hsi_normalized[:, :, selected_bands]
fig, axes = plt.subplots(1, N, figsize=(15, 5))
for i in range(N):
    ax = axes[i]
    im = ax.imshow(selected_data[:, :, i], cmap='gray')  # Display each PC
    ax.set_title(f'Selected {i+1}')
    fig.colorbar(im, ax=ax, orientation='vertical')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

#enhanced_bands = []
#for i in range(N):
  #  band = selected_data[:, :, i]
   # band_stretched = (band - np.min(band)) / (np.max(band) - np.min(band))
   # band_equalized = exposure.equalize_hist(band_stretched)
  #  enhanced_bands.append(band_equalized)
    
#fig, axes = plt.subplots(1, N, figsize=(15, 5))

#for i in range(N):
 #   ax = axes[i]
  #  im = ax.imshow(enhanced_bands[i], cmap='gray')  # Display each enhanced band
   # ax.set_title(f'Band {selected_bands[i]}')
   # fig.colorbar(im, ax=ax, orientation='vertical')

#plt.tight_layout()  # Adjust layout to prevent overlap
#plt.show()



selected_data = hsi_normalized[:, :, selected_bands]  # Extract the selected bands
combined_data = np.mean(selected_data, axis=2) 

combined_normalized = combined_data / np.max(combined_data) # Mean across the selected bands
#band_stretched = (combined_data - np.min(combined_data)) / (np.max(combined_data) - np.min(combined_data))
#band_equalized = exposure.equalize_hist(band_stretched)

plt.figure(figsize=(8, 8))
plt.imshow(combined_normalized, cmap='gray')  # Use 'gray' colormap for grayscale
plt.title('Combined Data (Mean of Selected Bands)')
plt.colorbar(label='Reflectance')
plt.show()


threshold = 0.25  # Example threshold
region_mask = combined_normalized > threshold

# Display the mask
plt.imshow(region_mask, cmap='gray')
plt.title('Region Mask (Thresholding)')
plt.colorbar()
plt.show()

masked_hsi = hsi_normalized  * region_mask[:, :, np.newaxis]
band_to_display = 152  # Change this to display a different band
plt.figure(figsize=(8, 8))
plt.imshow(masked_hsi[:, :, band_to_display], cmap='gray')
plt.title('Select ROIs: Draw a box for Normal and Anomalous Regions')
plt.colorbar(label='Reflectance')
plt.show()