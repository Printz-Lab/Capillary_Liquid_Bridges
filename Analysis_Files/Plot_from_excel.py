import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from matplotlib import cm

# --- Configuration ---
cfg = json.load(open(r"Analysis_Files\config.json"))
excel_path = Path(cfg["output_dir"]) / cfg["excel_output"]
force_keys = [
    'left_top_force_circle',
    'right_top_force_circle',
    'left_bottom_force_circle',
    'right_bottom_force_circle'
]

# --- Load Data ---
df_sep = pd.read_excel(excel_path, sheet_name="Plate_Separation")
df_forces = pd.read_excel(excel_path, sheet_name="Forces")

# Merge and clean
df = pd.merge(df_forces, df_sep[["frame", "plate_separation"]], on="frame")
df = df[df["frame"] >= 5].reset_index(drop=True)
df[force_keys] = df[force_keys].apply(pd.to_numeric, errors="coerce")
df["average_force"] = df[force_keys].mean(axis=1)
df["plate_separation_um"] = df["plate_separation"] * 1e3

# --- Compute Segments ---
sep = df["plate_separation"].values
direction = np.sign(np.diff(sep))
direction = np.insert(direction, 0, direction[0] if len(direction) > 0 else 0)
for i in range(1, len(direction)):
    if direction[i] == 0:
        direction[i] = direction[i-1]

segment_id = np.zeros_like(direction, dtype=int)
seg = 0
for i in range(1, len(direction)):
    if direction[i] != direction[i - 1]:
        seg += 1
    segment_id[i] = seg

df["segment_id"] = segment_id
df["direction"] = direction
n_segments = segment_id.max() + 1
cmap = cm.get_cmap('tab10', n_segments +1)

# --- Plot ---
plt.figure(figsize=(12, 7))
markers = ['o', 's', '^', 'D']

shown_labels = set()  # put this before the segment loop

for seg_id in range(n_segments):
    seg_mask = df["segment_id"] == seg_id
    direction_label = "Expansion" if df.loc[seg_mask, "direction"].iloc[0] > 0 else "Contraction"
    color = cmap(seg_id)

    # Plot individual forces (scatter)
    for i, key in enumerate(force_keys):
        label = key if key not in shown_labels else None
        plt.scatter(
            df.loc[seg_mask, "plate_separation_um"],
            df.loc[seg_mask, key],
            s=30,
            edgecolor='k',
            marker=markers[i],
            label=label,
            color=color,
        )
        shown_labels.add(key)


    # Plot average force
    plt.plot(
        df.loc[seg_mask, "plate_separation_um"],
        df.loc[seg_mask, "average_force"],
        label=f"Avg Force (Seg {seg_id+1}, {direction_label})",
        color=color,
        marker='x',
        linestyle='-'
    )

plt.xlabel("Plate Separation (μm)")
plt.ylabel("Force (μN)")
plt.title("Capillary Bridge Forces vs Plate Separation")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = Path(cfg["output_dir"]) / "capillary_bridge_forces_vs_plate_separation.png"
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()
