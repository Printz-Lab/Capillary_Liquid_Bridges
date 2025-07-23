#%%
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from Use_classifier import extract_features
from Substrate_tests.define_substrate_lines import interpolate_lines
def to_int(pt):
    return tuple(map(int, pt))

# Load classifiers
label_clf = joblib.load("mask_edge_classifier.pkl")
side_clf = joblib.load("contour_side_classifier.pkl")

# Paths
image_dir = Path(r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\My_better_CLB_videos\5-9\s1_tifs\moving_images")
mask_dir = Path(r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\masks_json")

image_paths = sorted(image_dir.glob("*.tif"))

# Manually defined substrate lines
bottom_line = ((89, 365), (925, 361))
top_start = ((87, 278), (913, 273))
top_end = ((134, 19), (818, 14))
top_lines = interpolate_lines(np.array(top_start[0]), np.array(top_start[1]),
                              np.array(top_end[0]), np.array(top_end[1]),
                              len(image_paths))


#%%
for idx, image_path in enumerate(image_paths):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    MAX_DIM = 1024
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    json_path = mask_dir / f"{image_path.stem}_masks.json"
    if not json_path.exists():
        continue

    with open(json_path, "r") as f:
        masks = json.load(f)

    features = []
    valid_indices = []
    for i, mask in enumerate(masks):
        feat = extract_features(mask, img.shape)
        if feat:
            features.append(feat)
            valid_indices.append(i)

    if not features:
        continue

    # Classify masks
    import pandas as pd
    X_pred = pd.DataFrame(features)
    edge_preds = label_clf.predict(X_pred)
    side_preds = side_clf.predict(X_pred[edge_preds == 1])

    left_mask = None
    right_mask = None
    for idx_mask, side in zip(np.where(edge_preds == 1)[0], side_preds):
        mask = masks[idx_mask]
        if side == "l":
            left_mask = mask
        elif side == "r":
            right_mask = mask

    # Draw overlay
    overlay = img.copy()
    colors = {"l": (0, 255, 0), "r": (255, 0, 0)}
    for side, mask in zip(["l", "r"], [left_mask, right_mask]):
        if mask:
            binary = np.array(mask["segmentation"]).astype(np.uint8)
            overlay[binary > 0] = (
                overlay[binary > 0] * 0.5 + np.array(colors[side]) * 0.5
            ).astype(np.uint8)

    # Draw bottom substrate
    cv2.line(overlay, bottom_line[0], bottom_line[1], (255, 255, 0), 5)
    # Draw interpolated top substrate
    top_line = top_lines[idx]
    cv2.line(overlay, tuple(top_line[0]), tuple(top_line[1]), (255, 0, 0), 5)

    # print(f"Image size: {img.shape}")
    # print(f"Bottom line: {bottom_line}")
    # print(f"Top line {idx}: {top_line}")
    # print("Drawing lines now...")

    cv2.circle(overlay, to_int(bottom_line[0]), 10, (0, 0, 255), -1)
    cv2.circle(overlay, to_int(bottom_line[1]), 10, (0, 0, 255), -1)
    cv2.circle(overlay, to_int(top_line[0]), 10, (0, 255, 255), -1)
    cv2.circle(overlay, to_int(top_line[1]), 10, (0, 255, 255), -1)


    plt.figure(figsize=(10, 6))
    plt.imshow(overlay)
    plt.title(f"Frame {idx}: SAM Masks + Substrate Lines")
    plt.axis("off")
    plt.show()

    # Optional: break here to test one frame
    break
