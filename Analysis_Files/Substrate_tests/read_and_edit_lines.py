import numpy as np
import cv2
import os
from pathlib import Path
from tkinter import filedialog, Tk
import sys
# Add parent directory to the module search path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import lines_file, image_dir

def get_line_from_user(image, window_name="Click two points"):
    points = []
    img_display = image.copy()

    def redraw():
        nonlocal img_display
        img_display = image.copy()
        for pt in points:
            cv2.circle(img_display, pt, 5, (0, 255, 0), -1)
        cv2.imshow(window_name, img_display)

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.setMouseCallback(window_name, click_event)
    redraw()

    while True:
        key = cv2.waitKey(0)
        if key in [13, 32]:  # Enter or space
            break
        elif key == ord('u') and points:
            points.pop()
            redraw()
        elif key == 27:  # Esc to cancel
            points = []
            break

    cv2.destroyWindow(window_name)
    return np.array(points) if len(points) == 2 else None



if __name__ == "__main__":
    folder, npz_path = image_dir, lines_file
    image_paths = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])
    image_paths = [os.path.join(folder, f) for f in image_paths]

    data = np.load(npz_path, allow_pickle=True)
    top_lines = list(data["top_lines"])
    bottom_lines = list(data["bottom_lines"])

    print(f"Loaded {len(image_paths)} images and {len(top_lines)} substrate lines")

    while True:
        try:
            frame_num = int(input(f"\nEnter frame number to update (0–{len(image_paths)-1}, or -1 to save and quit): "))
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if frame_num == -1:
            break
        if frame_num < 0 or frame_num >= len(image_paths):
            print("Frame out of range.")
            continue

        img = cv2.imread(image_paths[frame_num])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = 1024
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        print(f"Select TOP line for frame {frame_num}")
        top = get_line_from_user(img, f"Top Line Frame {frame_num}")
        if top is None:
            print("Top line not updated.")
        else:
            top_lines[frame_num] = (top[0].astype(int), top[1].astype(int))

        print(f"Select BOTTOM line for frame {frame_num}")
        bottom = get_line_from_user(img, f"Bottom Line Frame {frame_num}")
        if bottom is None:
            print("Bottom line not updated.")
        else:
            bottom_lines[frame_num] = (bottom[0].astype(int), bottom[1].astype(int))

    # --- Save updated lines ---
    save_path = Path(npz_path).with_name("updated_" + Path(npz_path).name)
    np.savez(
        save_path,
        top_lines=np.array(top_lines, dtype=object),
        bottom_lines=np.array(bottom_lines, dtype=object)
    )
    print(f"\n✅ Updated lines saved to: {save_path}")