import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import tkinter as tk
from tkinter import filedialog


def contour_viewer_with_dilation(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (7, 7), 2.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.25, bottom=0.3)

    ax.imshow(image_rgb)
    ax.axis("off")
    ax.set_title("Contours from Canny")

    # Slider axes
    ax_low = plt.axes([0.25, 0.2, 0.65, 0.03])
    ax_high = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_low = Slider(ax_low, 'Low Threshold', 0, 255, valinit=20, valstep=1)
    slider_high = Slider(ax_high, 'High Threshold', 0, 255, valinit=20, valstep=1)

    # Checkbox for dilation
    ax_check = plt.axes([0.025, 0.4, 0.15, 0.1])
    checkbox = CheckButtons(ax_check, ['Dilate'], [False])

    def update(val=None):
        low = int(slider_low.val)
        high = int(slider_high.val)
        use_dilate = checkbox.get_status()[0]

        edges = cv2.Canny(blurred, low, high)

        if use_dilate:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ax.clear()
        ax.imshow(image_rgb)
        for cnt in contours:
            cnt = cnt.squeeze()
            if len(cnt.shape) == 2 and cv2.contourArea(cnt) > 20:
                ax.plot(cnt[:, 0], cnt[:, 1], linewidth=1)
        ax.axis("off")
        ax.set_title(f"Contours (Low={low}, High={high}, Dilate={use_dilate})")
        fig.canvas.draw_idle()

    slider_low.on_changed(update)
    slider_high.on_changed(update)
    checkbox.on_clicked(update)

    update()
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
        contour_viewer_with_dilation(image_bgr)
    else:
        print("No file selected.")