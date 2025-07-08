import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
excel_path = r"Alannah_Sample_1\Alannah_S1_training.xlsx"
# force_keys = ['left_top_force', 'right_top_force', 'left_bottom_force', 'right_bottom_force']  # Change to any of:
force_keys = ['left_top_force_circle', 'right_top_force_circle', 'left_bottom_force_circle', 'right_bottom_force_circle']  # Or any other force keys like
<<<<<<< Updated upstream
# 'left_top_force', 'right_top_force', 'left_bottom_force', 'right_bottom_force',
# 'left_top_force_circle', etc.
=======

>>>>>>> Stashed changes
color_by = "frame"  # Could also color by plate separation if desired

# --- Load Data ---
df_sep = pd.read_excel(excel_path, sheet_name="Plate_Separation")
df_forces = pd.read_excel(excel_path, sheet_name="Forces")

# Merge on 'frame'
df = pd.merge(df_forces, df_sep[["frame", "plate_separation"]], on="frame")

average_forces = df[force_keys].mean(axis=1)


# --- Plot ---
plt.figure(figsize=(10, 6))
for i, force_key in enumerate(force_keys):
    print(df.columns)
    if force_key not in df.columns:
        print(f"Warning: {force_key} not found in DataFrame columns.")
        continue

    # Ensure the force column is numeric
    df[force_key] = pd.to_numeric(df[force_key], errors='coerce')
    # Drop NaNs
    df2 = df[["frame", force_key, "plate_separation"]].dropna()
    
    # Convert to microns for plotting
    separation_um = df2["plate_separation"] * 1e3
    forces = df2[force_key]
    frames = df2["frame"]
    markers = ['o', 's', '^', 'D']  # Different markers for each force type
    # Assign color based on frame value
    color = np.where(df2["frame"] < 30, 'tab:blue', 'tab:orange')
    sc =plt.scatter(separation_um, forces, c=color, s=30, edgecolor='k', marker=markers[i], label=force_key)


separation_um = df["plate_separation"] * 1e3  # Convert to microns
# Mask where frame < 30
mask_early = df["frame"] < 16

# Plot early segment
plt.plot(
    separation_um[mask_early],
    average_forces[mask_early],
    color='tab:blue',
    label='Average Force Expansion',
    marker='x'
)

# Plot late segment
plt.plot(
    separation_um[~mask_early],
    average_forces[~mask_early],
    color='tab:orange',
    label='Average Force Compression',
    marker='x'
)

plt.legend()
plt.xlabel("Plate Separation (mm)")
plt.ylabel("Force (μN)")
plt.title(f"{force_key} vs. Plate Separation")
plt.grid(True)
plt.tight_layout()
plt.show()
