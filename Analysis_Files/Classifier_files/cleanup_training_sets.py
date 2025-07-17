import os
import json
import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use("TkAgg")  # allows keypress interaction

# === CONFIG ===
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

csv_path = Path(cfg["output_dir"]) / "labeled_training_data_combined.csv"  # or "labeled_training_data_revised.csv"
classifier_path = label_clf

MAX_DIM = 1024  # image resize if large

# === Load ===
df = pd.read_csv(csv_path)
clf = joblib.load(classifier_path)

# Determine classifier type
is_side_classifier = "side" in df.columns
label_col = "side" if is_side_classifier else "label"
features = ["area", "perimeter", "aspect_ratio", "extent", "solidity", "centroid_x", "centroid_y"]

# Predict
preds = clf.predict(df[features])
df["predicted"] = preds
df["disagree"] = df["predicted"] != df[label_col]

# Filter disagreements
disagreements = df[df["disagree"] == True].reset_index(drop=True)
print(f"Found {len(disagreements)} possibly mislabeled entries")

# === Show each mismatch ===
def draw_mask_overlay(image, mask, color=(0, 255, 0)):
    overlay = image.copy()
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
    return overlay

def on_key(event):
    if event.key == 'q':
        plt.close("all")
        print("Exiting.")
        exit()

for i, row in disagreements.iterrows():
    image_path = Path(row["image_folder"]) / row["image"]
    json_path = Path(row["mask_folder"]) / f"{Path(row['image']).stem}_masks.json"
    if not image_path.exists() or not json_path.exists():
        print(f"Missing: {image_path} or {json_path}")
        continue

    # Load and resize image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    with open(json_path, "r") as f:
        masks = json.load(f)

    if row["mask_index"] >= len(masks):
        continue

    mask = masks[int(row["mask_index"])]
    display_img = draw_mask_overlay(image, mask)
    #just the most imediate folder (not path)
    folder = row["image_folder"].split(os.sep)[-1]
    # Show info
    fig, ax = plt.subplots()
    ax.imshow(display_img)
    ax.set_title(
        f"{folder} {row['image']} | Mask #{row['mask_index']} | True: {row[label_col]} | Pred: {row['predicted'] }"
    )
    ax.axis("off")
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
