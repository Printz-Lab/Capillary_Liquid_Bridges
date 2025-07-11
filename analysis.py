# %%
"""
interactive_analysis.py

A VSCode-friendly interactive script using cell markers.
Run cells in the VSCode Python Interactive window to:
  1. Define substrate lines via OpenCV windows
  2. Save interpolated lines
  3. Launch multiprocessing analysis for Excel & plot

In VSCode:
  - Open this file
  - Click “Run Cell” on each section (the `# %%` markers)
  - Make sure you have a GUI backend available (e.g. local machine)
"""

# %%
# 1️⃣ Setup: imports & path configuration
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import argparse

# add project root to path
top = Path(__file__).resolve().parent
sys.path.insert(0, str(top))

# import config values
from Analysis_Files.config import (
    image_dir as IMAGE_DIR,
    mask_dir as MASK_DIR,
    lines_file as LINES_FILE,
    excel_output as EXCEL_OUTPUT,
    debug_image_dir as DEBUG_IMAGE_DIR
)

# interpolation utility
from Analysis_Files.Substrate_tests.define_substrate_lines import load_images_from_folder, interpolate_lines

# %%
# 2️⃣ Cell magic: enable Qt for OpenCV windows
# Place this in a cell and run it once at start
# This magic works in VSCode Interactive/QtConsole, not in plain terminal
# If it errors, ensure your Python environment has Qt installed
from IPython.core.getipython import get_ipython
print(get_ipython())
get_ipython().run_line_magic('gui', 'qt')
cv2.startWindowThread()

# %%
# 3️⃣ Configuration: convert to Path and verify directories
image_dir       = Path(IMAGE_DIR)
mask_dir        = Path(MASK_DIR)
lines_file      = Path(LINES_FILE)
excel_output    = Path(EXCEL_OUTPUT)
debug_image_dir = Path(DEBUG_IMAGE_DIR)

# create output dir
debug_image_dir.mkdir(parents=True, exist_ok=True)

print(f"Images: {image_dir}\nMasks: {mask_dir}\nLines: {lines_file}\nExcel: {excel_output}")

# %%
# 4️⃣ Helper: capture two clicks via OpenCV

def get_line_via_cv(image, title="Select two points"):
    pts = []
    disp = image.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(title, disp)
            if len(pts) == 2:
                cv2.destroyWindow(title)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 800, 600)
    cv2.imshow(title, disp)
    cv2.setMouseCallback(title, click_event)

    # wait for two clicks
    while len(pts) < 2:
        cv2.waitKey(50)
    cv2.waitKey(1)
    return pts

# %%
# 5️⃣ Manual substrate-line picking
dir_paths = load_images_from_folder(str(image_dir))
num_frames = len(dir_paths)
step = 10  # sampling step
extra_frames = [0, 9, 10, 17, 20, 25, 30, 33]
frames_to_click = sorted(set(list(range(0, num_frames, step)) + extra_frames))
print("Frames to click:", frames_to_click)

# storage
top1, top2, bot1, bot2, idxs = [], [], [], [], []
for i in frames_to_click:
    img = cv2.imread(str(dir_paths[i]))
    pts = get_line_via_cv(img, title=f"Top — frame {i}")
    top1.append(pts[0]); top2.append(pts[1])
    pts = get_line_via_cv(img, title=f"Bottom — frame {i}")
    bot1.append(pts[0]); bot2.append(pts[1])
    idxs.append(i)

# %%
# 6️⃣ Interpolation & save
tops    = interpolate_lines(idxs, top1, top2, num_frames)
bottoms = interpolate_lines(idxs, bot1, bot2, num_frames)
np.savez(str(lines_file), top_lines=tops, bottom_lines=bottoms)
print(f"Saved substrate lines to {lines_file}")

# %%
# 7️⃣ Run full analysis
# This will spawn multiprocessing, write Excel & plot PNG
os.system(f"python plot_forces_multiprocessing.py")
print(f"Analysis complete. Excel: {excel_output}")

# %%
# 8️⃣ Display plot (if interactive window supports it)
from IPython.display import Image as Img
Img(str(image_dir.parent / excel_output.name.replace('.xlsx', '.png')))
