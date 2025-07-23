import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())

cfg = json.load(open(r"Analysis_Files\config.json"))
image_dir = Path(cfg["image_dir"])
mask_dir  = Path(cfg["mask_dir"])
output_dir = Path(cfg["output_dir"])
lines_file = Path(cfg["lines_file"])
label_clf_path = Path(cfg["label_clf_path"])
side_clf_path = Path(cfg["side_clf_path"])
excel_out = Path(cfg["output_dir"]) / cfg["excel_output"]
debug_dir = Path(cfg["output_dir"]) / cfg["debug_image_dir"]
first_frame_spacing = cfg["first_frame_spacing"]
sigma_surface_tension = cfg["sigma_surface_tension"]
# Load trained classifiers
label_clf = joblib.load(label_clf_path)
side_clf = joblib.load(side_clf_path)


os.makedirs(output_dir, exist_ok=True)

# Features to extract (same as in training)
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

    return [area, perimeter, aspect_ratio, extent, solidity, cx, cy]
if __name__ == "__main__":
    # Go through each image and predict left/right masks
    image_paths = sorted(
    list(Path(image_dir).glob("*.tif")) + list(Path(image_dir).glob("*.png"))
)
    print(f"Found {len(image_paths)} images to process.")
    for idx, image_path in enumerate(tqdm(image_paths)):
        json_path = Path(mask_dir) / f"{image_path.stem}_masks.json"
        if not json_path.exists():
            continue

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        MAX_DIM = 1024
        scale = MAX_DIM / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        with open(json_path, "r") as f:
            masks = json.load(f)

        features = []
        valid_indices = []
        for i, mask in enumerate(masks):
            feat = extract_features(mask, image.shape)
            if feat:
                features.append(feat)
                valid_indices.append(i)

        if not features:
            continue

        X_pred = pd.DataFrame(features, columns=["area", "perimeter", "aspect_ratio", "extent", "solidity", "centroid_x", "centroid_y"])
        print(f"Extracted features for {len(features)} masks in {image_path.name}")
        labels = label_clf.predict(X_pred)

        print(labels)
        edge_indices = [i for i, label in zip(valid_indices, labels) if label == 1]
        edge_features = X_pred.iloc[[valid_indices.index(i) for i in edge_indices]]

        print(f"Identified {len(edge_indices)} edge masks in {image_path.name}")
        side_preds = side_clf.predict(edge_features)

        # Overlay predictions
        overlay = image.copy()
        for idx_mask, side in zip(edge_indices, side_preds):
            color = (0, 255, 0) if side == "l" else (255, 0, 0)
            binary = np.array(masks[idx_mask]["segmentation"]).astype(np.uint8)
            overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        plt.imshow(overlay)
        plt.title(f"Predicted Left/Right Masks for {image_path.name}")
        plt.axis("off")
        out_file = Path(output_dir) / f"{image_path.stem}_prediction.png"
        plt.show()
        # plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)
        # plt.close()
