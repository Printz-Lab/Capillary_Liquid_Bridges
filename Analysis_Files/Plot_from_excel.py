import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from matplotlib import cm

import matplotlib as mpl

mpl.rcParams.update(
    {
        # 1) pick Arial for all sans-serif text…
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        # 2) make mathtext use Arial as well
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "rm",
        # 3) still your other style settings
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 14,
        "figure.figsize": (6, 8),
        "axes.linewidth": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "axes.grid": False,
        "savefig.dpi": 300,
        # if you had usetex on, turn it off so mathtext takes over:
        "text.usetex": False,
    }
)


# --- Configuration ---
cfg = json.load(open(r"Analysis_Files\config.json"))
excel_path = Path(cfg["output_dir"]) / cfg["excel_output"]
force_keys = [
    'left_top_force_circle',
    'right_top_force_circle',
    'left_bottom_force_circle',
    'right_bottom_force_circle'
    # 'left_top_force',
    # 'right_top_force',
    # 'left_bottom_force',
    # 'right_bottom_force'
]
labels = [
    'Left Top',
    'Right Top',
    'Left Bottom',
    'Right Bottom'
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
        label = labels[i] if key not in shown_labels else None
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
        linestyle='-'
    )

plt.xlabel("Plate Separation (μm)")
plt.ylabel("Force (μN)")
# plt.title("Capillary Bridge Forces vs Plate Separation")
plt.legend()
plt.grid(True)
plt.tight_layout()
save_path = Path(cfg["output_dir"]) / "capillary_bridge_forces_vs_plate_separation.png"
plt.savefig(save_path)
print(f"Plot saved to: {save_path}")
plt.show()
