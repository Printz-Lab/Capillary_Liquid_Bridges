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


# Load trained classifiers
label_clf = joblib.load("mask_edge_classifier2.pkl")
side_clf = joblib.load("contour_side_classifier2.pkl")

# Paths
image_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video"
mask_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video\masks_json"
output_dir = "predicted_mask_overlays"
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
        labels = label_clf.predict(X_pred)

        edge_indices = [i for i, label in zip(valid_indices, labels) if label == 1]
        edge_features = X_pred.iloc[[valid_indices.index(i) for i in edge_indices]]
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
