import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def test_watershed(image_bgr, thresh=0.4):
    original = image_bgr.copy()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Binary thresholding using Otsu
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to fill gaps
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Sure background (dilated binary)
    sure_bg = cv2.dilate(closed, kernel, iterations=3)

    # Distance transform to get sure foreground
    dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, thresh * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region = subtract fg from bg
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # So background is not 0, but 1
    markers[unknown == 255] = 0  # Mark unknown regions as 0

    # Apply watershed
    markers = cv2.watershed(image_bgr, markers)

    # Overlay boundaries on original image
    segmented = original.copy()
    segmented[markers == -1] = [0, 0, 255]  # Boundary marked by -1

    # Visualize all steps
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Original")
    axs[0, 1].imshow(binary, cmap='gray')
    axs[0, 1].set_title("Binary (Otsu)")
    axs[0, 2].imshow(dist_transform, cmap='jet')
    axs[0, 2].set_title("Distance Transform")

    axs[1, 0].imshow(sure_fg, cmap='gray')
    axs[1, 0].set_title("Sure Foreground")
    axs[1, 1].imshow(sure_bg, cmap='gray')
    axs[1, 1].set_title("Sure Background")
    axs[1, 2].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title("Watershed Segmentation")

    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create a simple GUI to select an image file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes("-topmost", True)  # Keep the file dialog on top
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif")],
    )

    if file_path:
        image_bgr = cv2.imread(file_path)
        roi_coords = cv2.selectROI("Select ROI", image_bgr, fromCenter=False, showCrosshair=True)
        if roi_coords is not None and roi_coords != (0, 0, 0, 0):
            x_min, y_min, width, height = roi_coords
            cropped = image_bgr[y_min:y_min + height, x_min:x_min + width]
            image_bgr = cropped
        else:
            print("No ROI selected, using full image.")
        if image_bgr is not None:
            for thresh in [0.3, .4, .5, .6, .7]:
                print(f"Testing with threshold: {thresh}") 
                test_watershed(image_bgr, thresh=thresh)
        else:
            print("Error: Could not read the image.")
    else:
        print("No file selected.")