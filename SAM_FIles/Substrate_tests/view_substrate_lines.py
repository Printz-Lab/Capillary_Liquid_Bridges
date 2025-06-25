import cv2
import numpy as np
import os
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
from pathlib import Path

# Add parent directory to the module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import lines_file, image_dir

def select_image_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with .tif Images")
    return folder_path

def load_images_from_folder(folder):
    images = []
    filenames = sorted([f for f in os.listdir(folder) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')])
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if h > 1024 or w > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        if img is not None:
            images.append((filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    return images

def draw_substrate_lines_from_npz(images, top_lines, bottom_lines):
    drawn_images = []
    for i, (name, img) in enumerate(images):
        img_copy = img.copy()

        if i >= len(top_lines) or i >= len(bottom_lines):
            print(f"Skipping frame {i}: line data out of bounds")
            continue

        top_line = top_lines[i]
        bottom_line = bottom_lines[i]

        if top_line is None or bottom_line is None:
            print(f"Skipping frame {i}: missing line data")
            continue

        try:
            cv2.line(img_copy, tuple(top_line[0]), tuple(top_line[1]), (255, 0, 0), 2)       # Top = red
            cv2.line(img_copy, tuple(bottom_line[0]), tuple(bottom_line[1]), (0, 255, 0), 2)  # Bottom = green
        except Exception as e:
            print(f"Error drawing lines for frame {i}: {e}")
            continue

        drawn_images.append((name, img_copy))
    return drawn_images


def view_images_with_slider(drawn_images):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    img_display = ax.imshow(drawn_images[0][1])
    ax.set_title(drawn_images[0][0])
    ax.axis('off')

    ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(drawn_images)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        name, img = drawn_images[idx]
        img_display.set_data(img)
        ax.set_title(name)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    folder = image_dir
    npz_path = lines_file

    if not npz_path:
        print("No .npz file selected, exiting.")
        exit()

    data = np.load(npz_path, allow_pickle=True)
    top_lines = data["top_lines"]
    bottom_lines = data["bottom_lines"]

    images = load_images_from_folder(folder)
    drawn = draw_substrate_lines_from_npz(images, top_lines, bottom_lines)
    view_images_with_slider(drawn)
