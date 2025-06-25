from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
from pathlib import Path
from config import (
    excel_output, debug_image_dir
)

if __name__ == "__main__":
    debug_dir = Path(debug_image_dir)
    debug_dir.mkdir(exist_ok=True)

    import multiprocessing
    from Final_RunFile_Working import (
        process_frame_with_ellipses,
        image_dir, mask_dir,
        bottom_lines,
        top_lines,
        pixel_to_meter,
        interpolate_lines,
        process_single_frame
    )

    image_paths = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.png"))
    

    args_list = [
    (idx, image_path, mask_dir, bottom_lines, top_lines, pixel_to_meter, debug_dir)
    for idx, image_path in enumerate(image_paths)
    ]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_single_frame, args_list), total=len(image_paths)))

    import pandas as pd

    # Initialize lists of rows for each category
    forces_data = []
    angles_data = []
    positions_data = []
    curvature_data = []
    plate_separation_data = []

    for idx, result in enumerate(results):
        if result is None:
            continue
        
        base = {"frame": idx}

        # Plate separation
        plate_separation_data.append({
            **base,
            "plate_separation": result.get("plate_separation_m"),
        })
        # Forces
        forces_data.append({
            **base,
            "left_top_force": result.get("left_top_force"),
            "left_bottom_force": result.get("left_bottom_force"),
            "right_top_force": result.get("right_top_force"),
            "right_bottom_force": result.get("right_bottom_force"),
            "right_top_force_circle": result.get("right_top_force_circle"),
            "right_bottom_force_circle": result.get("right_bottom_force_circle"),
            "left_top_force_circle": result.get("left_top_force_circle"),
            "left_bottom_force_circle": result.get("left_bottom_force_circle")
        })

        # Angles
        angles_data.append({
            **base,
            "left_top_angle": result.get("left_top_angle"),
            "left_bottom_angle": result.get("left_bottom_angle"),
            "right_top_angle": result.get("right_top_angle"),
            "right_bottom_angle": result.get("right_bottom_angle")
        })

        # Contact Positions
        positions_data.append({
            **base,
            "left_top_X": result.get("left_top_X"),
            "left_top_Y": result.get("left_top_Y"),
            "left_bottom_X": result.get("left_bottom_X"),
            "left_bottom_Y": result.get("left_bottom_Y"),
            "right_top_X": result.get("right_top_X"),
            "right_top_Y": result.get("right_top_Y"),
            "right_bottom_X": result.get("right_bottom_X"),
            "right_bottom_Y": result.get("right_bottom_Y")
        })

        # Curvature + neck width
        curvature_data.append({
            **base,
            "H_mean_left": result.get("H_mean_left"),
            "H_mean_right": result.get("H_mean_right"),
            "Y_star": result.get("Y*")
        })

    
    with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:
        pd.DataFrame(forces_data).to_excel(writer, sheet_name="Forces", index=False)
        pd.DataFrame(angles_data).to_excel(writer, sheet_name="Angles", index=False)
        pd.DataFrame(positions_data).to_excel(writer, sheet_name="Contact_Positions", index=False)
        pd.DataFrame(curvature_data).to_excel(writer, sheet_name="Curvature_Y", index=False)
        pd.DataFrame(plate_separation_data).to_excel(writer, sheet_name="Plate_Separation", index=False)

    print(f"\n✅ Results saved to: {excel_output}")


    # Filter valid results
    frame_indices = []
    plate_separations = []
    f_lt, f_rt, f_lb, f_rb = [], [], [], []
    f_lt_circle, f_rt_circle, f_lb_circle, f_rb_circle = [], [], [], []
    for idx, result in enumerate(results):
        if result:
            plate_separations.append(result.get("plate_separation_m"))
            frame_indices.append(idx)
            f_lt.append(result.get("left_top_force"))
            f_rt.append(result.get("right_top_force"))
            f_lb.append(result.get("left_bottom_force"))
            f_rb.append(result.get("right_bottom_force"))
            f_lt_circle.append(result.get("left_top_force_circle"))
            f_rt_circle.append(result.get("right_top_force_circle"))
            f_lb_circle.append(result.get("left_bottom_force_circle"))
            f_rb_circle.append(result.get("right_bottom_force_circle"))

    

    # print(type(f_lt), type(f_rt), type(f_lb), type(f_rb))
    # Convert forces to numpy arrays and take absolute values
    f_lt_clean = np.array([abs(x) if x is not None else np.nan for x in f_lt])
    f_lt = np.abs(f_lt_clean)
    f_rt_clean = np.array([abs(x) if x is not None else np.nan for x in f_rt])
    f_rt = np.abs(f_rt_clean)
    f_lb_clean = np.array([abs(x) if x is not None else np.nan for x in f_lb])
    f_lb = np.abs(f_lb_clean)
    f_rb_clean = np.array([abs(x) if x is not None else np.nan for x in f_rb])
    f_rb = np.abs(f_rb_clean)

    # Convert forces to numpy arrays and take absolute values for circles
    f_lt_circle_clean = np.array([abs(x) if x is not None else np.nan for x in f_lt_circle])
    f_lt_circle = np.abs(f_lt_circle_clean)
    f_rt_circle_clean = np.array([abs(x) if x is not None else np.nan for x in f_rt_circle])
    f_rt_circle = np.abs(f_rt_circle_clean)
    f_lb_circle_clean = np.array([abs(x) if x is not None else np.nan for x in f_lb_circle])
    f_lb_circle = np.abs(f_lb_circle_clean)
    f_rb_circle_clean = np.array([abs(x) if x is not None else np.nan for x in f_rb_circle])
    f_rb_circle = np.abs(f_rb_circle_clean)

    #average the lt, rt, lb, rb forces together
    Average = np.nanmean([f_lt, f_rt, f_lb, f_rb], axis=0)
    Average_circle = np.nanmean([f_lt_circle, f_rt_circle, f_lb_circle, f_rb_circle], axis=0)

    # Convert plate separations to mm
    plate_separations = np.array(plate_separations) * 1000  # Convert to mm

    # Plot results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    # plt.plot(plate_separations, f_lt, label="Left Top Force")
    # plt.plot(plate_separations, f_rt, label="Right Top Force")
    # plt.plot(plate_separations, f_lb, label="Left Bottom Force")
    # plt.plot(plate_separations, f_rb, label="Right Bottom Force")
    # plt.plot(plate_separations, Average, label="Average Force", linestyle=':', color='orange')
    plt.plot(plate_separations, f_lt_circle, label="Left Top Force Circle", linestyle='--')
    plt.plot(plate_separations, f_rt_circle, label="Right Top Force Circle", linestyle='--')
    plt.plot(plate_separations, f_lb_circle, label="Left Bottom Force Circle", linestyle='--')
    plt.plot(plate_separations, f_rb_circle, label="Right Bottom Force Circle", linestyle='--')
    plt.plot(plate_separations, Average_circle, label="Average Force Circle", linestyle=':', color='orange')
  
    plt.xlabel("Plate_Separation (mm)")
    plt.ylabel("Force (μN)")
    plt.title("Capillary Bridge Forces vs Frame Number")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("capillary_bridge_forces.png")
    plt.show()
