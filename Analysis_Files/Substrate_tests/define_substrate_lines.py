import cv2
import numpy as np
import os
from pathlib import Path
from tkinter import filedialog, Tk
import argparse

def select_image_folder():
    root = Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select Folder with Images")
    return folder_path

def load_images_from_folder(folder_path, extensions={".tif", ".jpg", ".png"}):
    return sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in extensions
    ])

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
        if key == 13 or key == 32:  # Enter or Space = confirm
            break
        elif key == ord('u') and points:
            points.pop()
            redraw()
        elif key == 27:  # Esc to cancel
            points = []
            break

    cv2.destroyWindow(window_name)

    if len(points) != 2:
        raise ValueError("Exactly two points must be selected. (Esc to cancel, 'u' to undo)")

    return np.array(points[0]), np.array(points[1])


def interpolate_lines(frames, pt1_list, pt2_list, num_frames):
    full_lines = [None] * num_frames
    for i in range(len(frames) - 1):
        f_start, f_end = frames[i], frames[i + 1]
        for j in range(f_end - f_start + 1):
            alpha = j / (f_end - f_start)
            pt1 = (1 - alpha) * pt1_list[i] + alpha * pt1_list[i + 1]
            pt2 = (1 - alpha) * pt2_list[i] + alpha * pt2_list[i + 1]
            full_lines[f_start + j] = (pt1.astype(int), pt2.astype(int))
    return full_lines

if __name__ == "__main__":
    # folder = select_image_folder()
    folder = argparse
    image_paths = load_images_from_folder(folder)

    # --- Frame selection configuration ---
    step = 10  # Select every 10th frame
    custom_frames = [0, 9,17,25 , 33]  # Add any specific frames (like max separation point)

    # Combine and sort unique frame indices
    selected_indices = sorted(set(list(range(0, len(image_paths), step)) + custom_frames))
    selected_image_paths = [image_paths[i] for i in selected_indices]

    print(f"Selected frames: {selected_indices}")


    top_pts1, top_pts2 = [], []
    bot_pts1, bot_pts2 = [], []
    frame_indices = []

    for i in range(0, len(selected_image_paths)):
        img = cv2.imread(selected_image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = 1024 / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
        print(f"Select TOP line for frame {i}")
        top1, top2 = get_line_from_user(img, f"Top Line Frame {i}")
        print(f"Select BOTTOM line for frame {i}")
        bot1, bot2 = get_line_from_user(img, f"Bottom Line Frame {i}")
        top_pts1.append(top1)
        top_pts2.append(top2)
        bot_pts1.append(bot1)
        bot_pts2.append(bot2)
        frame_indices.append(selected_indices[i])


    top_lines = interpolate_lines(frame_indices, top_pts1, top_pts2, len(image_paths))
    bottom_lines = interpolate_lines(frame_indices, bot_pts1, bot_pts2, len(image_paths))

    savefile = "Alannah_S1_substrates.npz"

    np.savez(
    savefile,
    top_lines=np.array(top_lines, dtype=object),
    bottom_lines=np.array(bottom_lines, dtype=object)
)

    print(f"Saved top_lines and bottom_lines to {savefile}")
