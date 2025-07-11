import os

# config.py

# Paths
# image_dir = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\My_better_CLB_videos\5-9\s1_tifs\moving_images"
image_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video"
# mask_dir = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\masks_json"
mask_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video\masks_json"
output_dir = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\output"
lines_file = r"C:\Users\raglo\OneDrive - University of Arizona\Documents\GitHub\Capillary_Liquid_Bridges\Alannah_S1_substrates.npz"
# lines_file = r"C:\Users\raglo\OneDrive - University of Arizona\Documents\GitHub\Capillary_Liquid_Bridges\shifted_updated_interpolated_substrate_lines.npz"
label_clf_path = "mask_edge_classifier2.pkl"
side_clf_path = "contour_side_classifier2.pkl"            
excel_output = "Alannah_S1_training.xlsx"
debug_image_dir = excel_output.replace(".xlsx", "_debug_images")

# Ensure output_dir exists
os.makedirs(output_dir, exist_ok=True)

# Update output file paths to be under output_dir
excel_output = os.path.join(output_dir, excel_output)
debug_image_dir = os.path.join(output_dir, debug_image_dir)

first_frame_spacing = 0.1e-3 # mm
sigma_surface_tension = 72  # mN/m

