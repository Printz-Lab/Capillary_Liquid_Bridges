import os
import json
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd

from Final_RunFile_Working import (
    process_frame_with_ellipses,
    interpolate_lines,
    bottom_line, top_start, top_end,
    label_clf, side_clf,
    image_dir, mask_dir, pixel_to_meter
)

# Ensure paths and classifier variables are loaded correctly from Final_RunFile_Working.py
image_paths = sorted(image_dir.glob("*.tif"))
top_lines = interpolate_lines(
    np.array(top_start[0]), np.array(top_start[1]),
    np.array(top_end[0]), np.array(top_end[1]),
    len(image_paths)
)

# Containers for data
frame_indices = []
forces_left_top = []
forces_right_top = []
forces_left_bottom = []
forces_right_bottom = []

# Loop over all frames
for idx, image_path in tqdm(enumerate(image_paths), total =len(image_paths), desc="Processing frames"):
    json_path = mask_dir / f"{image_path.stem}_masks.json"
    if not json_path.exists():
        continue

    with open(json_path, "r") as f:
        masks = json.load(f)

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    top_line = top_lines[idx]

    result = process_frame_with_ellipses(img, masks, bottom_line, top_line)
    if result:
        frame_indices.append(idx)
        forces_left_top.append(result.get("left_top_force"))
        forces_right_top.append(result.get("right_top_force"))
        forces_left_bottom.append(result.get("left_bottom_force"))
        forces_right_bottom.append(result.get("right_bottom_force"))
    # Save results to CSV
    results_df = pd.DataFrame({
        "frame_index": frame_indices,
        "left_top_force": forces_left_top,
        "right_top_force": forces_right_top,
        "left_bottom_force": forces_left_bottom,
        "right_bottom_force": forces_right_bottom
    })
    csv_path = Path("forces_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path.resolve()}")
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(frame_indices, forces_left_top, label="Left Top Force")
plt.plot(frame_indices, forces_right_top, label="Right Top Force")
plt.plot(frame_indices, forces_left_bottom, label="Left Bottom Force")
plt.plot(frame_indices, forces_right_bottom, label="Right Bottom Force")
plt.xlabel("Frame Index")
plt.ylabel("Force (μN)")
plt.title("Capillary Bridge Forces vs Frame Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
