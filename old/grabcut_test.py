import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

print("Threads OpenCV is using:", cv2.getNumThreads())

def run_grabcut(image_bgr, rect):
    """
    Perform GrabCut segmentation given an image and bounding box.
    Parameters:
        image_bgr: Input BGR image (OpenCV format)
        rect: Tuple (x, y, width, height) defining bounding box
    Returns:
        segmented image with background removed
    """
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Apply GrabCut
    cv2.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    # Convert result to binary mask
    bin_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image_bgr * bin_mask[:, :, np.newaxis]

    return segmented, bin_mask


def show_grabcut_result(image_bgr, rect):
    segmented, bin_mask = run_grabcut(image_bgr, rect)

    # Plot result
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].add_patch(plt.Rectangle(
        (rect[0], rect[1]), rect[2], rect[3],
        edgecolor='lime', facecolor='none', linewidth=2
    ))
    axs[1].imshow(bin_mask, cmap='gray')
    axs[1].set_title("GrabCut Mask")
    axs[2].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Segmented Droplet")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a simple GUI to select an image file
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")],
    )

    if file_path:
        image_bgr = cv2.imread(file_path)
        if image_bgr is None:
            print("Error loading image.")
        else:
            # Define a bounding box (x, y, width, height)
            roi_coords = cv2.selectROI("Select ROI", image_bgr, fromCenter=False, showCrosshair=True)
            if roi_coords is not None and roi_coords != (0, 0, 0, 0):
                rect = (int(roi_coords[0]), int(roi_coords[1]), int(roi_coords[2]), int(roi_coords[3]))
            else:
                print("No ROI selected, using default rectangle.")
                # Default rectangle for demonstration
            show_grabcut_result(image_bgr, rect)
    else:
        print("No file selected.")
