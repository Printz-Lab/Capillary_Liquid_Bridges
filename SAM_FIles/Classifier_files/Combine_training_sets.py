import pandas as pd

# File paths
raw1 = "labeled_training_data_combined.csv"
raw2 = "labeled_training_data_3.csv"
merged_raw_out = "labeled_training_data_alannah.csv"
image_folder1 = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\My_better_CLB_videos\5-9\s1_tifs"
image_folder2 = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video"
mask_folder1 = r"C:\Users\raglo\OneDrive - University of Arizona\Printz Lab\Data\Capillary_Bridges\masks_json"
mask_folder2 = r"D:\Capillary_bridging_data_alannah\Trial 1 (water on glass)\video\masks_json"

revised1 = "labeled_training_data_revised_combined.csv"
revised2 = "labeled_training_data_revised_3.csv"
merged_revised_out = "labeled_training_data_revised_alannah.csv"

def load_and_tag(path, img_dir, mask_dir):
    df = pd.read_csv(path)
    df["image_folder"] = img_dir
    df["mask_folder"] = mask_dir
    return df

# Raw (edge) classifier data
df_raw1 = pd.read_csv(raw1)
df_raw2 = load_and_tag(raw2, image_folder2, mask_folder2)
df_raw_combined = pd.concat([df_raw1, df_raw2], ignore_index=True)
df_raw_combined.to_csv(merged_raw_out, index=False)
print(f"✅ Merged edge classifier data → {merged_raw_out} ({len(df_raw_combined)} rows)")

# Revised (side) classifier data
df_rev1 = pd.read_csv(revised1)
df_rev2 = load_and_tag(revised2, image_folder2, mask_folder2)
df_rev_combined = pd.concat([df_rev1, df_rev2], ignore_index=True)
df_rev_combined.to_csv(merged_revised_out, index=False)
print(f"✅ Merged side classifier data → {merged_revised_out} ({len(df_rev_combined)} rows)")