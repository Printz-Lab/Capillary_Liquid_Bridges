import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")  # Enables interactive window for key press detection

# CONFIGURATION
image_dir =r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video"  # Folder with .tif files
mask_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video\masks_json"  # Folder with .json files
input_csv = "labeled_training_data_3.csv"
output_csv = "labeled_training_data_revised_3.csv"

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
    cx = np.mean(np.where(binary)[1]) / image_shape[1]
    cy = np.mean(np.where(binary)[0]) / image_shape[0]

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "centroid_x": cx,
        "centroid_y": cy,
    }

label_result = {"value": None, "side": None}

def on_key(event):
    if event.key in ('0', '1', 'q', 'l', 'r', 'b', 's'):
        label_result["value"] = event.key
        plt.close()

def show_mask(image, mask, idx, label):
    overlay = image.copy()
    color = (0, 255, 0)
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    title = f"Mask #{idx} - L=left, R=right, B=bad label, S=skip, Q=quit (Current: {label})"
    ax.set_title(title)
    ax.axis("off")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# === Load labeled CSV ===
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"CSV not found: {input_csv}")

df = pd.read_csv(input_csv)
df = df[df["label"] == 1]

new_data = []

for i, row in df.iterrows():
    image_path = Path(image_dir) / row["image"]
    json_path = Path(mask_dir) / (Path(row["image"]).stem + "_masks.json")

    if not image_path.exists() or not json_path.exists():
        print(f"Missing image or mask: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    MAX_DIM = 1024
    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    with open(json_path, "r") as f:
        masks = json.load(f)

    mask_idx = row["mask_index"]
    if mask_idx >= len(masks):
        continue

    mask = masks[mask_idx]
    features = extract_features(mask, image.shape)
    if features is None:
        continue

    label_result["value"] = None
    show_mask(image, mask, mask_idx, label=row.get("side", "None"))
    label = label_result["value"]

    if label == "q":
        pd.DataFrame(new_data).to_csv(output_csv, index=False)
        print("Saved and exiting...")
        exit(0)
    elif label == "s":
        continue
    elif label in ("l", "r"):
        features.update(row.to_dict())
        features["side"] = label
        new_data.append(features)
    elif label == "b":  # Bad label
        features.update(row.to_dict())
        features["label"] = 0
        features["side"] = ""
        new_data.append(features)

pd.DataFrame(new_data).to_csv(output_csv, index=False)
print(f"Done. Saved {len(new_data)} revised examples to {output_csv}")
