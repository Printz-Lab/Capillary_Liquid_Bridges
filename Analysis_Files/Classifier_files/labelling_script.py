import os
import json
from json import JSONDecodeError
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")  # Enables interactive window for key press detection
import sys
# CONFIGURATION
sys.path.append(os.getcwd())

cfg = json.load(open(r"Analysis_Files\config.json"))
image_dir = Path(cfg["image_dir"])
mask_dir  = Path(cfg["mask_dir"])
output_dir = Path(cfg["output_dir"])
lines_file = Path(cfg["lines_file"])
label_clf = Path(cfg["label_clf_path"])
side_clf  = Path(cfg["side_clf_path"])
excel_out = Path(cfg["output_dir"]) / cfg["excel_output"]
debug_dir = Path(cfg["output_dir"]) / cfg["debug_image_dir"]
first_frame_spacing = cfg["first_frame_spacing"]
sigma_surface_tension = cfg["sigma_surface_tension"]

# CONFIGURATION
output_csv = Path(cfg["output_dir"]) / "labeled_training_data.csv"

AREA_THRESHOLD = 1000  # Masks with area smaller than this will be auto-labeled 0 and skipped from GUI

# Create output if not exists
labeled_data = []

# === Helper Functions ===
def extract_features(mask, image_shape):
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0] if contours else None
    if cnt is None:
        return None

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h if h > 0 else 0
    extent = area / (w * h) if w * h > 0 else 0
    hull = cv2.convexHull(cnt)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    cx = np.mean(np.where(binary)[1]) / image_shape[1]  # normalized
    cy = np.mean(np.where(binary)[0]) / image_shape[0]  # normalized

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "centroid_x": cx,
        "centroid_y": cy,
    }

label_result = {"value": None}

def on_key(event):
    if event.key in ('0', '1', 's', 'q'):
        label_result["value"] = event.key
        plt.close()

def show_mask(image, mask, idx):
    overlay = image.copy()
    color = (0, 255, 0)
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    ax.set_title(f"Mask #{idx} - Press 1=edge, 0=not, s=skip, q=quit")
    ax.axis("off")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# === Loop Over All Files ===
image_paths = sorted(
    list(Path(image_dir).glob("*.tif")) + list(Path(image_dir).glob("*.png"))
)

for i, image_path in enumerate(image_paths):
    if not i%7 == 0:
        continue
    json_path = Path(mask_dir) / (image_path.stem + "_masks.json")
    if not json_path.exists():
        print(f"No mask file for {image_path.name}")
        continue

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    MAX_DIM = 1024

    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        with open(json_path, "r") as f:
            masks = json.load(f)
    except JSONDecodeError as e:
        print(f"⚠️  Skipping {json_path.name}: JSON decode error at {e.pos} – {e.msg}")
        continue

    for idx, mask in enumerate(masks):
        features = extract_features(mask, image.shape)
        if features is None:
            continue
        
        if features["area"] < AREA_THRESHOLD:
            features["label"] = 0
            features["image"] = image_path.name
            features["mask_index"] = idx
            labeled_data.append(features)
            continue

        label_result["value"] = None
        show_mask(image, mask, idx)
        label = label_result["value"]

        if label == "q":
            pd.DataFrame(labeled_data).to_csv(output_csv, index=False)
            print("Saved and exiting...")
            exit(0)
        elif label == "s":
            continue
        elif label in ("0", "1"):
            features["label"] = int(label)
            features["image"] = image_path.name
            features["mask_index"] = idx
            labeled_data.append(features)

# === Save after finishing ===
pd.DataFrame(labeled_data).to_csv(output_csv, index=False)
print(f"Done. Saved {len(labeled_data)} labeled examples to {output_csv}")
