import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import tkinter as tk
from tkinter import filedialog
from cv2ellipse import *

def adaptive_threshold_with_contours_viewer(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.3)

    # Initial parameters
    block_size = 31
    C = 3

    ax.imshow(gray, cmap='gray')
    ax.set_title(f"Adaptive Threshold with Contours")
    ax.axis("off")

    # Sliders
    ax_block = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_C = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_block = Slider(ax_block, 'Block Size', 3, 51, valinit=block_size, valstep=2)
    slider_C = Slider(ax_C, 'C (Offset)', -20, 20, valinit=C, valstep=1)

    # Radio buttons for method
    ax_method = plt.axes([0.025, 0.4, 0.2, 0.15])
    radio = RadioButtons(ax_method, ['mean', 'gaussian'])

    def update(val=None):
        blk = int(slider_block.val)
        c = int(slider_C.val)
        method = radio.value_selected

        # Adaptive thresholding
        thresh_type = cv2.ADAPTIVE_THRESH_MEAN_C if method == 'mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        binary = cv2.adaptiveThreshold(
            gray, 255, thresh_type,
            cv2.THRESH_BINARY_INV, blk, c
        )

        # Find contours on binary mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw threshold and contours
        ax.clear()
        ax.imshow(binary, cmap='gray')
        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt.shape) == 2 and cv2.contourArea(cnt) > 20:
                ax.plot(cnt[:, 0], cnt[:, 1], linewidth=1, color='lime')
        ax.set_title(f"Adaptive ({method}) - Block={blk}, C={c}, Contours={len(contours)}")
        ax.axis("off")
        fig.canvas.draw_idle()

    # Bind events
    slider_block.on_changed(update)
    slider_C.on_changed(update)
    radio.on_clicked(update)

    update()
    plt.show()

if __name__ == "__main__":
    # Create a simple GUI to select an image file
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp, *.tif")],
    )

    if file_path:
        image_bgr = cv2.imread(file_path)
        roi_coords = select_roi(image_bgr)
        if roi_coords is not None:
            x_min, y_min, x_max, y_max = roi_coords
            cropped = image_bgr[y_min:y_max, x_min:x_max]
        else:
            print("No ROI selected, using full image.")

        if cropped is not None:
            adaptive_threshold_with_contours_viewer(cropped)
        else:
            print("Error loading image.")
    else:
        print("No file selected.")