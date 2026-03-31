# label_masks.py
import os
import sys
import json
from json import JSONDecodeError
from pathlib import Path
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl

# --- Backend setup (must happen BEFORE importing pyplot) ---
def setup_backend(backend: str):
    """
    Choose a Matplotlib backend BEFORE importing pyplot.
    backend options:
      - 'auto'   : widget in IPython if available; else do nothing
      - 'widget' : ipympl Jupyter widget backend
      - 'tk'     : TkAgg desktop windows
      - 'inline' : inline static images
    """
    backend = (backend or "auto").lower()

    def in_ipython():
        try:
            from IPython import get_ipython  # noqa
            return get_ipython() is not None
        except Exception:
            return False

    if backend == "auto":
        if in_ipython():
            # Prefer ipympl if installed; otherwise keep whatever the notebook already has
            try:
                import ipympl  # noqa: F401
                mpl.use("module://ipympl.backend_nbagg")
            except Exception:
                pass
        else:
            # Non-notebook: use TkAgg for interactive key events
            mpl.use("TkAgg")
    elif backend == "widget":
        mpl.use("module://ipympl.backend_nbagg")
    elif backend == "tk":
        mpl.use("TkAgg")
    elif backend == "inline":
        # Inline is usually set via magic, but this backend works too
        mpl.use("module://matplotlib_inline.backend_inline")
    else:
        raise ValueError(f"Unknown backend: {backend}")

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Label masks with keyboard input.")
    p.add_argument("--config", default="Analysis_Files/config.json", help="Path to config.json")
    p.add_argument("--every-n", type=int, default=7, help="Process every Nth image (default: 7)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "widget", "tk", "inline"],
                   help="Matplotlib backend selection (default: auto)")
    return p.parse_args(argv)

# We set backend before importing pyplot
def import_pyplot():
    import matplotlib.pyplot as plt
    return plt

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
    ys, xs = np.where(binary)
    cx = float(xs.mean()) / image_shape[1] if xs.size else 0.0  # normalized
    cy = float(ys.mean()) / image_shape[0] if ys.size else 0.0  # normalized

    return {
        "area": area,
        "perimeter": perimeter,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "solidity": solidity,
        "centroid_x": cx,
        "centroid_y": cy,
    }

def run_labeler(config_path="Analysis_Files/config.json", every_n=7, backend="auto"):
    setup_backend(backend)
    plt = import_pyplot()

    # CONFIGURATION
    cfg = json.load(open(config_path, "r"))
    image_dir = Path(cfg["image_dir"])
    mask_dir  = Path(cfg["mask_dir"])
    output_dir = Path(cfg["output_dir"])
    lines_file = Path(cfg["lines_file"])
    label_clf = Path(cfg["label_clf_path"])
    side_clf  = Path(cfg["side_clf_path"])
    excel_out = output_dir / cfg["excel_output"]
    debug_dir = output_dir / cfg["debug_image_dir"]
    first_frame_spacing = cfg["first_frame_spacing"]
    sigma_surface_tension = cfg["sigma_surface_tension"]

    output_csv = output_dir / "labeled_training_data.csv"
    AREA_THRESHOLD = 1000  # Masks with area smaller than this are auto-labeled 0

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    labeled_data = []
    label_result = {"value": None}

    def on_key(event):
        if event.key in ("0", "1", "s", "q"):
            label_result["value"] = event.key
            plt.close()

    def show_mask(image, mask, idx):
        overlay = image.copy()
        color = (0, 255, 0)
        binary = np.array(mask["segmentation"]).astype(np.uint8)
        overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        fig, ax = plt.subplots()
        ax.imshow(overlay)
        ax.set_title(f"Mask #{idx} - Press 1=edge, 0=not, s=skip, q=quit")
        ax.axis("off")
        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show()

    image_paths = sorted(list(image_dir.glob("*.tif")) + list(image_dir.glob("*.png")))
    MAX_DIM = 1024

    for i, image_path in enumerate(image_paths):
        if every_n and (i % every_n != 0):
            continue

        json_path = mask_dir / f"{image_path.stem}_masks.json"
        if not json_path.exists():
            print(f"No mask file for {image_path.name}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"⚠️  Could not read image: {image_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        scale = MAX_DIM / max(h, w)
        if scale < 1:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        try:
            with open(json_path, "r") as f:
                masks = json.load(f)
        except JSONDecodeError as e:
            print(f"⚠️  Skipping {json_path.name}: JSON decode error at {e.pos} – {e.msg}")
            continue

        for idx, mask in enumerate(masks):
            features = extract_features(mask, image.shape)
            if features is None:
                continue

            # Small masks get auto-labeled 0
            if features["area"] < AREA_THRESHOLD:
                features["label"] = 0
                features["image"] = image_path.name
                features["mask_index"] = idx
                labeled_data.append(features)
                continue

            label_result["value"] = None
            show_mask(image, mask, idx)
            label = label_result["value"]

            if label == "q":
                pd.DataFrame(labeled_data).to_csv(output_csv, index=False)
                print(f"Saved {len(labeled_data)} rows and exiting...")
                sys.exit(0)
            elif label == "s":
                continue
            elif label in ("0", "1"):
                features["label"] = int(label)
                features["image"] = image_path.name
                features["mask_index"] = idx
                labeled_data.append(features)

    pd.DataFrame(labeled_data).to_csv(output_csv, index=False)
    print(f"Done. Saved {len(labeled_data)} labeled examples to {output_csv}")

def main(argv=None):
    args = parse_args(argv)
    # Avoid messing with sys.path; rely on config paths
    run_labeler(config_path=args.config, every_n=args.every_n, backend=args.backend)

if __name__ == "__main__":
    main()
