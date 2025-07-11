import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Load and prepare data ---
raw_df = pd.read_csv(r"C:\Users\raglo\OneDrive - University of Arizona\Documents\GitHub\Capillary_Liquid_Bridges\labeled_training_data_alannah.csv")  # Full dataset for edge detection
revised_df = pd.read_csv(r"C:\Users\raglo\OneDrive - University of Arizona\Documents\GitHub\Capillary_Liquid_Bridges\labeled_training_data_revised_alannah.csv")  # Revised dataset for side classification

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

joblib.dump(label_clf, "mask_edge_classifier2.pkl")
print("Edge classifier saved to mask_edge_classifier.pkl")

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

joblib.dump(side_clf, "contour_side_classifier2.pkl")
print("Side classifier saved to contour_side_classifier.pkl")
