import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage import exposure, filters


def calibrate_hsi(original, white, dark):
    #hsi_data = original.load()
    #white_ref_data = white.load()
    #dark_ref_data = dark.load()
    calibrated_hsi = (original - dark) / (white - dark + 1e-6)
    calibrated_hsi = np.nan_to_num(calibrated_hsi, nan=0.0, posinf=0.0, neginf=0.0)
    return calibrated_hsi

def display_band(hsi, band):
    plt.imshow(hsi[:, :, band], cmap='gray')
    plt.title('First Band of Calibrated HSI')
    plt.colorbar(label='Reflectance')
    plt.show()

#Wavelengths
wavelengths = [
    400.00, 401.34, 402.68, 404.02, 405.36, 406.70, 408.04, 409.38, 410.71, 412.05, 413.39, 414.73, 416.07, 417.41, 418.75, 420.09, 421.43, 422.77, 424.11, 425.45, 426.79, 428.12, 429.46, 430.80, 432.14, 433.48, 434.82, 
    436.16, 437.50, 438.84, 440.18, 441.52, 442.86, 444.20, 445.54, 446.88, 448.21, 449.55, 450.89, 452.23, 453.57, 454.91, 456.25, 457.59, 458.93, 460.27, 461.61, 462.95, 464.29, 465.62, 466.96, 468.30, 469.64, 470.98, 
    472.32, 473.66, 475.00, 476.34, 477.68, 479.02, 480.36, 481.70, 483.04, 484.38, 485.71, 487.05, 488.39, 489.73, 491.07, 492.41, 493.75, 495.09, 496.43, 497.77, 499.11, 500.45, 501.79, 503.12, 504.46, 505.80, 507.14, 
    508.48, 509.82, 511.16, 512.50, 513.84, 515.18, 516.52, 517.86, 519.20, 520.54, 521.88, 523.21, 524.55, 525.89, 527.23, 528.57, 529.91, 531.25, 532.59, 533.93, 535.27, 536.61, 537.95, 539.29, 540.62, 541.96, 543.30, 
    544.64, 545.98, 547.32, 548.66, 550.00, 551.34, 552.68, 554.02, 555.36, 556.70, 558.04, 559.38, 560.71, 562.05, 563.39, 564.73, 566.07, 567.41, 568.75, 570.09, 571.43, 572.77, 574.11, 575.45, 576.79, 578.12, 579.46, 
    580.80, 582.14, 583.48, 584.82, 586.16, 587.50, 588.84, 590.18, 591.52, 592.86, 594.20, 595.54, 596.88, 598.21, 599.55, 600.89, 602.23, 603.57, 604.91, 606.25, 607.59, 608.93, 610.27, 611.61, 612.95, 614.29, 615.62, 
    616.96, 618.30, 619.64, 620.98, 622.32, 623.66, 625.00, 626.34, 627.68, 629.02, 630.36, 631.70, 633.04, 634.38, 635.71, 637.05, 638.39, 639.73, 641.07, 642.41, 643.75, 645.09, 646.43, 647.77, 649.11, 650.45, 651.79, 
    653.12, 654.46, 655.80, 657.14, 658.48, 659.82, 661.16, 662.50, 663.84, 665.18, 666.52, 667.86, 669.20, 670.54, 671.88, 673.21, 674.55, 675.89, 677.23, 678.57, 679.91, 681.25, 682.59, 683.93, 685.27, 686.61, 687.95, 
    689.29, 690.62, 691.96, 693.30, 694.64, 695.98, 697.32, 698.66, 700.00, 701.34, 702.68, 704.02, 705.36, 706.70, 708.04, 709.38, 710.71, 712.05, 713.39, 714.73, 716.07, 717.41, 718.75, 720.09, 721.43, 722.77, 724.11, 
    725.45, 726.79, 728.12, 729.46, 730.80, 732.14, 733.48, 734.82, 736.16, 737.50, 738.84, 740.18, 741.52, 742.86, 744.20, 745.54, 746.88, 748.21, 749.55, 750.89, 752.23, 753.57, 754.91, 756.25, 757.59, 758.93, 760.27, 
    761.61, 762.95, 764.29, 765.62, 766.96, 768.30, 769.64, 770.98, 772.32, 773.66, 775.00, 776.34, 777.68, 779.02, 780.36, 781.70, 783.04, 784.38, 785.71, 787.05, 788.39, 789.73, 791.07, 792.41, 793.75, 795.09, 796.43, 
    797.77, 799.11, 800.45, 801.79, 803.12, 804.46, 805.80, 807.14, 808.48, 809.82, 811.16, 812.50, 813.84, 815.18, 816.52, 817.86, 819.20, 820.54, 821.88, 823.21, 824.55, 825.89, 827.23, 828.57, 829.91, 831.25, 832.59, 
    833.93, 835.27, 836.61, 837.95, 839.29, 840.62, 841.96, 843.30, 844.64, 845.98, 847.32, 848.66, 850.00, 851.34, 852.68, 854.02, 855.36, 856.70, 858.04, 859.38, 860.71, 862.05, 863.39, 864.73, 866.07, 867.41, 868.75, 
    870.09, 871.43, 872.77, 874.11, 875.45, 876.79, 878.12, 879.46, 880.80, 882.14, 883.48, 884.82, 886.16, 887.50, 888.84, 890.18, 891.52, 892.86, 894.20, 895.54, 896.88, 898.21, 899.55, 900.89, 902.23, 903.57, 904.91, 
    906.25, 907.59, 908.93, 910.27, 911.61, 912.95, 914.29, 915.62, 916.96, 918.30, 919.64, 920.98, 922.32, 923.66, 925.00, 926.34, 927.68, 929.02, 930.36, 931.70, 933.04, 934.38, 935.71, 937.05, 938.39, 939.73, 941.07, 
    942.41, 943.75, 945.09, 946.43, 947.77, 949.11, 950.45, 951.79, 953.12, 954.46, 955.80, 957.14, 958.48, 959.82, 961.16, 962.50, 963.84, 965.18, 966.52, 967.86, 969.20, 970.54, 971.88, 973.21, 974.55, 975.89, 977.23, 
    978.57, 979.91, 981.25, 982.59, 983.93, 985.27, 986.61, 987.95, 989.29, 990.62, 991.96, 993.30, 994.64, 995.98, 997.32, 998.66
]



# Load the hyperspectral image
#hsi = sp.open_image("/workspace/band_selection/anom_2_2024-01-26_15-34-16/capture/anom_2_2024-01-26_15-34-16.hdr")
hsi = sp.open_image('/workspace/src/dataset/hsi/anom_1_2024-01-26_15-21-55/capture/anom_1_2024-01-26_15-21-55.hdr')

# Load the white and dark references
white_ref = sp.open_image("/workspace/src/dataset/hsi/white_ref_2024-01-26_16-00-30/capture/white_ref_2024-01-26_16-00-30.hdr")
dark_ref = sp.open_image("/workspace/src/dataset/hsi/dark_ref_shutter_cap_on/capture/dark_ref_shutter_2024-01-26_16-03-56.hdr")

#Loading
hsi_data = hsi.load()
white_ref_data = white_ref.load()
drk_ref_data = dark_ref.load()
hsi_data = calibrate_hsi(hsi_data, white_ref_data, drk_ref_data)

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

    # Plot the selected ROIs on a different image
    fig, ax = plt.subplots()
    ax.imshow(hsi_data[:, :, 152], cmap='gray')  # Display the first band of the hyperspectral data

    # Draw bounding boxes around the selected ROIs
    rect_normal = plt.Rectangle((x1_n, y1_n), x2_n - x1_n, y2_n - y1_n, edgecolor='green', linewidth=2, fill=False)
    rect_anomalous = plt.Rectangle((x1_a, y1_a), x2_a - x1_a, y2_a - y1_a, edgecolor='red', linewidth=2, fill=False)

    ax.add_patch(rect_normal)
    ax.add_patch(rect_anomalous)

    plt.title('Selected ROIs with Bounding Boxes')
    plt.show()

    # Plot the mean spectra
    mean_spectrum_split = np.mean(anomalous_region, axis=(0, 1))
    mean_spectrum_normal = np.mean(normal_region, axis=(0, 1))

    plt.plot(wavelengths, mean_spectrum_split, label='Anomalous Region')
    plt.plot(wavelengths, mean_spectrum_normal, label='Normal Region')
    plt.xlabel('Wavelength')
    plt.ylabel('Reflectance')
    plt.title('Mean Spectra of Selected Regions')
    plt.legend()
    plt.show()
else:
    print("ROI selection was not completed.")

# Statistically. 
band_differences = np.abs(mean_spectrum_split - mean_spectrum_normal)
N = 5 # Number of bands to select
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


threshold = 0.09  # Example threshold
region_mask = combined_normalized < threshold

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
