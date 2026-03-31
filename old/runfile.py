# %%

import tkinter as tk
from tkinter import filedialog
import os
import cv2
import numpy as np
from SAM_FIles.cv2ellipse import *
from SAM_FIles.forces import *

# Hide the root Tk window

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
# Ask user to pick a folder
image_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.bmp")],
)
if not image_path:
    print("No file selected.")
    exit()


results = []

# Process images
print(f"\nProcessing {os.path.basename(image_path)}…")
result = process_droplet_two_lobes(image_path, roi_type="auto")
if not result:
    print(f"stopping {os.path.basename(image_path)} due to processing failure.")
    exit()

# Unpack results
ellipse_left, ellipse_right, (x_min, y_min, x_max, y_max), image, cropped = result
roi_height, roi_width = y_max - y_min, x_max - x_min

# Contact angle analysis
left_contacts = extract_all_contact_angles(
    ellipse_left, 0, cropped.shape[0] - 1, "left"
)
left_meriodonal = get_meriodonal_profile(
    ellipse_left, left_contacts, "left"
)
right_contacts = extract_all_contact_angles(
    ellipse_right, right_contacts, "right"
)
right_meriodonal = get_meriodonal_profile(
    ellipse_right, 0, cropped.shape[0] - 1, "right"
)

draw_meriodonal_profile(
    full_image= image.copy, roi_coords=(x_min, y_min, x_max, y_max), profile_points = left_meriodonal, side ="left"
)
draw_meriodonal_profile(
    full_image= image.copy, roi_coords=(x_min, y_min, x_max, y_max), profile_points = right_meriodonal, side ="right"
)

print(f"Left contacts: {left_contacts}")
print(f"Right contacts: {right_contacts}")
# Geometric analysis
origin, l_pt, r_pt, min_dist = compute_curve_distance(ellipse_left, ellipse_right)
# %%
if not origin:
    print("distance calculation failed.")
    exit()

data = {
    "filename": os.path.basename(image_path),
    "left_top_angle": None,
    "left_top_Y": None,
    "left_top_X": None,
    "left_bottom_angle": None,
    "left_bottom_Y": None,
    "left_bottom_X": None,
    "right_top_angle": None,
    "right_top_Y": None,
    "right_top_X": None,
    "right_bottom_angle": None,
    "right_bottom_Y": None,
    "right_bottom_X": None,
    "y*": None,
}
Ystar = min_dist / 2 if min_dist else None
data["y*"] = Ystar


# Process contact points and angles
def process_contacts(contacts, side, origin):
    for pos in ["top", "bottom"]:
        contact = contacts.get(pos, {})
        if contact["point"] is not None and contact["angle_deg"] is not None:
            # Transform coordinates
            Xn, Yn = transform_point_to_frame(contact["point"], origin)
            # Store values with absolute angle
            data[f"{side}_{pos}_angle"] = abs(contact["angle_deg"])
            data[f"{side}_{pos}_Y"] = Yn
            data[f"{side}_{pos}_X"] = Xn


if origin:
    process_contacts(left_contacts, "left", origin)
    process_contacts(right_contacts, "right", origin)

results.append(data)


# Transform coordinates
pts_l = ellipse_to_points(*ellipse_left)
pts_r = ellipse_to_points(*ellipse_right)
# Xl, Yl = transform_points_to_new_frame(pts_l[:,0], pts_l[:,1], origin)
# Xr, Yr = transform_points_to_new_frame(pts_r[:,0], pts_r[:,1], origin)
# %%

# Generate debug visualization
debug_img = image.copy()

# Draw fitted ellipses on full image
if ellipse_left is not None:
    (lx, ly), (lma, lmi), lang = ellipse_left
    cv2.ellipse(
        debug_img,
        (int(lx + x_min), int(ly + y_min)),
        (int(lma / 2), int(lmi / 2)),
        lang,
        0,
        360,
        (0, 0, 255),
        2,
    )

if ellipse_right is not None:
    (rx, ry), (rma, rmi), rang = ellipse_right
    cv2.ellipse(
        debug_img,
        (int(rx + x_min), int(ry + y_min)),
        (int(rma / 2), int(rmi / 2)),
        rang,
        0,
        360,
        (255, 0, 0),
        2,
    )

# Draw coordinate axes
origin_full = (int(origin[0] + x_min), int(origin[1] + y_min))
cv2.line(debug_img, (origin_full[0], y_min), (origin_full[0], y_max), (0, 255, 255), 2)
cv2.line(debug_img, (x_min, origin_full[1]), (x_max, origin_full[1]), (0, 255, 255), 2)

# Draw contact points and angles
debug_img = draw_contact_angle_debug(
    debug_img, left_contacts["top"], (x_min, y_min), (255, 0, 0), 0, "L-T: "
)
debug_img = draw_contact_angle_debug(
    debug_img, left_contacts["bottom"], (x_min, y_min), (255, 0, 0), 0, "L-B: "
)
debug_img = draw_contact_angle_debug(
    debug_img, right_contacts["top"], (x_min, y_min), (0, 255, 0), 0, "R-T: "
)
debug_img = draw_contact_angle_debug(
    debug_img, right_contacts["bottom"], (x_min, y_min), (0, 255, 0), 0, "R-B: "
)

# Show results
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
plt.title("Analysis Results - Fitted Ellipses and Contact Points")
plt.axis("off")
plt.show()

# Comprehensive printout
print(f"\n=== Results for {os.path.basename(image_path)} ===")
# transform contact points for below


# Contact points and angles
print("\nIntersection Points and Angles:")
for side, contacts in [("Left", left_contacts), ("Right", right_contacts)]:
    for pos in ["Top", "Bottom"]:
        contact = contacts[pos.lower()]
        pt = contact["point"]
        angle = contact["angle_deg"]

        print(f"\n{side} {pos} Contact:")

        # Handle coordinates
        if pt is not None:
            try:
                x_roi, y_roi = pt  # Explicit unpacking
                Xn, Yn = transform_point_to_frame((x_roi, y_roi), origin)

                # print(f"  Image Coordinates: ({x_roi+x_min:.1f}, {y_roi+y_min:.1f})")
                print(f"  Transformed System: (X={Xn:.1f}, Y={Yn:.1f})")

            except (ValueError, TypeError) as e:
                print(f"  Transformation error: {str(e)}")

        if angle is not None:
            absangle = abs(angle)
            print(f"  Angle: {absangle:+.1f}°")
        else:
            print("  Angle: Could not be calculated")


print("\nGeometric Analysis:")
print(f"Y* (neck width): {Ystar:.2f} px")
print(f"Image origin: ({origin_full[0]:.1f}, {origin_full[1]:.1f})")

###############################################################################################################################

# Solve for a and b using both contact points' equations


# Known values from image analysis (using absolute values for angles)
for side in ["left", "right"]:
    y_star = data["y*"]
    yc1 = np.abs(data[f"{side}_top_Y"])
    yc2 = np.abs(data[f"{side}_bottom_Y"])
    theta1 = np.deg2rad(
        abs(data[f"{side}_top_angle"])
    )  # Absolute value for angle direction
    theta2 = np.deg2rad(abs(data[f"{side}_bottom_angle"]))

    # After contact angle calculation
    bridge_type = classify_bridge(y_star, yc1, yc2, np.radians(theta1), np.radians(theta2))
    a, b = calculate_parameters(
        bridge_type, y_star, yc1, yc2, np.radians(theta1), np.radians(theta2)
    )
    print(a, b)
    H = get_H_from_nodoid_or_unduloid(a, bridge_type)
    
    # Get experimental points for error cal pfig, ax = plt.subplots(figsize=(12, 8))r ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))man processing loop:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Store all plottable artists
    # In your main processing loop:
    # In your plotting section:
    artists = []
    theory_line = draw_theoretical_curve(
        ax, bridge_type, a, b, origin, (x_min, y_min, x_max, y_max), side
    )

    if theory_line is not None:
        # Add theoretical curve
        color = "red" if side == "left" else "blue"
        theory_line = ax.plot(
            theory_line[1],
            theory_line[0],
            c=color,
            lw=2,
            label=f"Theoretical {bridge_type.capitalize()} {side.capitalize()}",
        )
        artists.append(theory_line[0])


    # Create legend only if we have entries
    if artists:
        ax.legend(handles=artists)
    else:
        ax.text(
            0.5,
            0.5,
            "No Valid Theoretical Curve",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    plt.show()
    # After parameter calculation
    print(f"Bridge type: {bridge_type}")
    print(f"Parameters: a={a:.2f}, b={b:.2f}")
    print(f"ROI coords: x={x_min}-{x_max}, y={y_min}-{y_max}")
    print(f"Origin point: {origin_full}")
    # %%

