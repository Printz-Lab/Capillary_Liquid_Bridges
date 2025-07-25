import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path
import json
import os

# --- Load and prepare data ---
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

raw_df = pd.read_csv(Path(cfg["output_dir"]) / "labeled_training_data.csv")  # Full dataset for edge detection
revised_df = pd.read_csv(Path(cfg["output_dir"]) / "labeled_training_data_LR.csv")  # Revised dataset for side classification

# --- Stage 1: Train edge (0=not edge, 1=edge) classifier on raw data ---
label_df = raw_df.copy()
label_non_features = ["image", "mask_index", "side", "image_folder", "mask_folder"]
label_features = [col for col in label_df.columns if col not in label_non_features + ["label"]]
X_label = label_df[label_features]
y_label = label_df["label"]

X_label_train, X_label_test, y_label_train, y_label_test = train_test_split(X_label, y_label, test_size=0.2, random_state=42)

label_clf = RandomForestClassifier(n_estimators=100, random_state=42)
label_clf.fit(X_label_train, y_label_train)

print("Label Classification Report:\n")
y_label_pred = label_clf.predict(X_label_test)
print(classification_report(y_label_test, y_label_pred))
clf_path_dir = label_clf_path.parent
label_clf_path = clf_path_dir / "mask_edge_classifier6.pkl"
joblib.dump(label_clf, label_clf_path)
print(f"Edge classifier saved to {label_clf_path}")

# --- Stage 2: Train side classifier for edge masks (1=valid edge) on revised data ---
side_df = revised_df[revised_df["label"] == 1].copy()
side_non_features = ["image", "mask_index", "label", "image_folder", "mask_folder"]
side_features = [col for col in side_df.columns if col not in side_non_features + ["side"]]
X_side = side_df[side_features]
y_side = side_df["side"]

X_side_train, X_side_test, y_side_train, y_side_test = train_test_split(X_side, y_side, test_size=0.2, random_state=42)

side_clf = RandomForestClassifier(n_estimators=100, random_state=42)
side_clf.fit(X_side_train, y_side_train)

print("Side Classification Report:\n")
y_side_pred = side_clf.predict(X_side_test)
print(classification_report(y_side_test, y_side_pred))
side_clf_path_dir = side_clf_path.parent
side_clf_path = side_clf_path_dir / "contour_side_classifier6.pkl"
joblib.dump(side_clf, side_clf_path)
print(f"Side classifier saved to {side_clf_path}")
