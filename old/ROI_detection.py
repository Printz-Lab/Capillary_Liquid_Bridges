import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import pandas as pd

# --- Configuration ---
image_path = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\My_better_CLB_videos\5-9\s1_tifs\frame_0052.tif"
mask_path = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\masks_json\frame_0052_masks.json"

# --- Load image and masks ---
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
MAX_DIM = 1024  # or 768, depending on your GPU

# Resize image if too large
h, w = image.shape[:2]
scale = MAX_DIM / max(h, w)
if scale < 1:
    image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

edge_clf_path = "mask_edge_classifier.pkl"
side_clf_path = "contour_side_classifier.pkl"

# === Load classifiers ===
edge_clf = joblib.load(edge_clf_path)
side_clf = joblib.load(side_clf_path)

# === Helper: extract features ===
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

with open(mask_path, "r") as f:
    masks = json.load(f)

# === Extract features and classify ===
features_list = []
valid_masks = []

for i, mask in enumerate(masks):
    features = extract_features(mask, image.shape)
    if features:
        features_list.append(features)
        valid_masks.append((i, mask))

if not features_list:
    raise RuntimeError("No valid masks with extractable features.")

features_df = pd.DataFrame(features_list)

# Step 1: Use edge classifier
edge_preds = edge_clf.predict(features_df)
edge_indices = [i for i, p in zip(range(len(valid_masks)), edge_preds) if p == 1]

# Step 2: Use side classifier on edge masks only
edge_features_df = features_df.iloc[edge_indices]
side_preds = side_clf.predict(edge_features_df)

# Organize predicted masks
left_mask = None
right_mask = None

for idx, side in zip(edge_indices, side_preds):
    mask = valid_masks[idx][1]
    if side == "l":
        left_mask = mask
    elif side == "r":
        right_mask = mask

# === Plot the selected masks and substrate ===
overlay = image.copy()
colors = {"left": (0, 255, 0), "right": (255, 0, 0)}

for label, mask in [("left", left_mask), ("right", right_mask)]:
    if mask is None:
        continue
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(colors[label]) * 0.5).astype(np.uint8)

# Estimate substrate line (based on max y of left/right masks)
substrate_y_max_vals = []
substrate_y_min_vals = []
for m in [left_mask, right_mask]:
    if m:
        bin_mask = np.array(m["segmentation"]).astype(np.uint8)
        ys = np.where(bin_mask)[0]
        if len(ys) > 0:
            substrate_y_max_vals.append(np.percentile(ys, 99))  # Use 90th percentile to reduce outliers
            substrate_y_min_vals.append(np.percentile(ys, 1))

if len(substrate_y_max_vals) == 2:
    substrate_y_max = int(np.mean(substrate_y_max_vals))
    cv2.line(overlay, (0, substrate_y_max), (overlay.shape[1], substrate_y_max), (255, 0, 0), 2)
if len(substrate_y_min_vals) == 2:
    substrate_y_min = int(np.mean(substrate_y_min_vals))
    cv2.line(overlay, (0, substrate_y_min), (overlay.shape[1], substrate_y_min), (0, 0, 255), 2)


# === Show result ===
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.title("Left & Right Masks with Estimated Substrate Line")
plt.axis("off")
plt.show()
