import os
import json
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

# --- Configuration ---
mask_dir = Path(r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\masks_json")  # Update this
area_threshold = 1000  # Minimum area in pixels

def mask_area(mask):
    binary = np.array(mask["segmentation"]).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0] if contours else None
    return cv2.contourArea(cnt) if cnt is not None else 0

json_files = list(mask_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

for json_file in tqdm(json_files, desc="Pruning masks"):
    with open(json_file, "r") as f:
        masks = json.load(f)

    original_size = os.path.getsize(json_file)
    # Filter masks
    filtered = [m for m in masks if mask_area(m) >= area_threshold]

    # Overwrite with reduced file
    with open(json_file, "w") as f:
        json.dump(filtered, f)

    # Optional: print file stats
    
    new_size = os.path.getsize(json_file)
    print(f"{json_file.name}: {len(filtered)} masks, {new_size/1024:.1f} KB")
    print(f"Reduced from {original_size/1024:.1f} KB to {new_size/1024:.1f} KB")

print("✅ Finished pruning small masks.")
