import matplotlib.pyplot as plt
import numpy as np
import logging
import spectral 
import matplotlib.pyplot as plt
from spectral import get_rgb
from ultralytics import YOLO
import numpy as np



# Load the model
model = YOLO("/workspace/src/preprocess/YOLOv8_trained.pt")

imagefile = "/workspace/src/Season_4/Normal/rgb/s1_norm1.png"

# Iterate over each image file

    # Perform prediction
segmentation = model.predict(imagefile, save=False, save_txt=False, box=True, imgsz=640, line_thickness=1, retina_masks=True)


# Extract original image, masks, and boxes
orig_img = segmentation[0].orig_img
masks = segmentation[0].masks
boxes = segmentation[0].boxes
orig_img = orig_img[..., ::-1]  # 


if masks is not None:
    mask_data = masks.data.cpu().numpy()
    box_data = boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    
    # Create an empty mask image
    mask_image = np.zeros_like(orig_img)

    # Color for visualization (e.g., red with some transparency)
    color = [0, 0, 1]  # Red color in RGB

    for i in range(mask_data.shape[0]):
        # If the mask is valid, overlay it on the mask image
        if mask_data[i].sum() > 0:  # Check if the mask has any pixels
            mask_image[mask_data[i] > 0] = color  # Set the mask color

    # Overlay the mask on the original image
    overlay_img = np.where(mask_image == 0, orig_img, mask_image * 0.5 + orig_img* 0.5)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay_img)  # Show the overlaid image
    for box in box_data:
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                            edgecolor='blue', linewidth=2, fill=False))
    plt.title('Segmented Image with Bounding Boxes and Masks')
    plt.axis('off')
    plt.show()

print("Processing completed.")
