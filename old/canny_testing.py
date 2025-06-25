import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import filedialog

def canny_edge_viewer(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 1.5)

    # Create initial Canny edge map
    def update(val):
        low = int(slider_low.val)
        high = int(slider_high.val)
        edges = cv2.Canny(blurred, low, high)
        coords = np.column_stack(np.where(edges > 0))

        ax.clear()
        ax.imshow(image_rgb)
        if len(coords) > 0:
            ax.plot(coords[:, 1], coords[:, 0], 'r.', markersize=0.8)
        ax.set_title(f'Canny Edges: Low={low}, High={high}')
        ax.axis("off")
        fig.canvas.draw_idle()

    # Initial thresholds
    low_init = 50
    high_init = 150

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.imshow(image_rgb)
    ax.set_title(f'Canny Edges: Low={low_init}, High={high_init}')
    ax.axis("off")

    # Add sliders
    ax_low = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_high = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_low = Slider(ax_low, 'Low Threshold', 0, 255, valinit=low_init, valstep=1)
    slider_high = Slider(ax_high, 'High Threshold', 0, 255, valinit=high_init, valstep=1)

    slider_low.on_changed(update)
    slider_high.on_changed(update)

    update(None)  # draw initial
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
        if image_bgr is not None:
            canny_edge_viewer(image_bgr)
        else:
            print("Error loading image.")
    else:
        print("No file selected.")

    