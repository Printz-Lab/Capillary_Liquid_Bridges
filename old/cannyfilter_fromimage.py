import cv2
import os
import glob
import numpy as np
# Define the input folder and output folder
input_folder = r"C:\Users\ezrap\OneDrive\Documents\Spring 2025 HW\Printz Lab Research\Capillary Bridging\4-15\s1\selected"  # Replace with the path to your folder containing .tif files
output_folder = r"C:\Users\ezrap\OneDrive\Documents\Spring 2025 HW\Printz Lab Research\Capillary Bridging\Anton's snapshots\APTMS Top" # Replace with the folder where you want to save the edge images

# Ensure the output directory exists, create if not
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all .tif files in the input folder
tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif"))) #change whether its tif or png or jpg

# Parameters for Canny edge detection
t_lower = 150     # Lower Threshold
t_upper = 230  # Upper Threshold

# Process each image in the folder
for tif_file in tif_files:
    # Read the image
    img = cv2.imread(tif_file, cv2.IMREAD_GRAYSCALE)  # Ensure image is read in grayscale
    
    if img is None:
        print(f"Error reading image: {tif_file}")
        continue
    
    # Apply the Canny Edge filter
    edge = cv2.Canny(img, t_lower, t_upper)

    # Display the original and edge images (optional)
    cv2.imshow('Original', img)
    cv2.imshow('Edge', edge)
    cv2.waitKey(1)  # Wait for a short time (you can adjust or remove this as needed)

    # Save the edge-detected image to the output folder
    output_filename = os.path.join(output_folder, os.path.basename(tif_file))  # Keep the same filename
    cv2.imwrite(output_filename, edge)

    print(f"Saved edge image to: {output_filename}")

cv2.destroyAllWindows()