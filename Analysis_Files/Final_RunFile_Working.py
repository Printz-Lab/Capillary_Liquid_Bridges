# %%
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from Classifier_files.Use_classifier import extract_features
from pprint import pprint
from Substrate_tests.define_substrate_lines import interpolate_lines
from forces import (
    numerical_kappa,
    calculate_mean_curvature,
    calculate_force,
    new_curvature_calculation,
    calculate_total_force_with_circle_model,
)
from cv2ellipse import (
    ellipse_to_points,
    fit_ellipse_to_contour,
    extract_all_contact_angles,
    get_meriodonal_profile,
    get_meridonal_profile_new,
    transform_points_to_new_frame,
    transform_point_to_frame,
    draw_contact_angle_debug,
    compute_curve_distance,
    fit_circle_to_contour_near_y,
)
import json 

cfg = json.load(open(r"Analysis_Files\config.json"))
image_dir = Path(cfg["image_dir"])
mask_dir  = Path(cfg["mask_dir"])
output_dir = Path(cfg["output_dir"])
lines_file = Path(cfg["lines_file"])
label_clf_path = Path(cfg["label_clf_path"])
side_clf_path = Path(cfg["side_clf_path"])
excel_out = Path(cfg["output_dir"]) / cfg["excel_output"]
debug_dir = Path(cfg["output_dir"]) / cfg["debug_image_dir"]
first_frame_spacing = cfg["first_frame_spacing"]
sigma_surface_tension = cfg["sigma_surface_tension"]




def to_int(pt):
    return tuple(map(int, pt))


# Load classifiers
label_clf = joblib.load(label_clf_path)
side_clf = joblib.load(side_clf_path)


image_dir = Path(image_dir)
mask_dir = Path(mask_dir)
image_paths = sorted(image_dir.glob("*.tif"))

# read substrate lines from file
lines = np.load(lines_file, allow_pickle=True)
top_lines = lines["top_lines"]
bottom_lines = lines["bottom_lines"]

pixel_to_meter = first_frame_spacing / np.linalg.norm(
    np.array(top_lines[0][1]) - np.array(bottom_lines[0][1])
)


# %% Main function for processing a frame
def process_frame_with_ellipses(img, masks, bottom_line, top_line):
    results = {}

    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    features = []
    valid_indices = []
    for i, mask in enumerate(masks):
        feat = extract_features(mask, img.shape)
        if feat:
            features.append(feat)
            valid_indices.append(i)
    print(f"Valid indices: {valid_indices}")
    if not features:
        return None

    import pandas as pd
    from matplotlib import cm

    X_pred = pd.DataFrame(
        features,
        columns=[
            "area",
            "perimeter",
            "aspect_ratio",
            "extent",
            "solidity",
            "centroid_x",
            "centroid_y",
        ],
    )
    edge_preds = label_clf.predict(X_pred)
    side_preds = side_clf.predict(X_pred[edge_preds == 1])

    left_mask = right_mask = None
    for idx_mask, side in zip(np.where(edge_preds == 1)[0], side_preds):
        mask = masks[idx_mask]
        if side == "l":
            left_mask = mask
        elif side == "r":
            right_mask = mask

    def get_contour(mask, x_min, x_max):
        binary = np.array(mask["segmentation"]).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        contour = contour.reshape(-1, 2)
        mask = (contour[:, 0] > x_min+5) & (contour[:, 0] < x_max-5)
        cropped = contour[mask]
        return cropped.reshape(-1, 1, 2) if len(cropped) > 0 else None

    x_min_left = min(int(bottom_line[0][0]), int(top_line[0][0]))-2
    x_max_left = np.mean([int(bottom_line[0][0]), int(bottom_line[1][0])])
    x_min_right = np.mean([int(bottom_line[0][0]), int(bottom_line[1][0])])
    x_max_right = max(int(bottom_line[1][0]), int(top_line[1][0]))+8

    contour_left = (
        get_contour(left_mask, x_min=x_min_left, x_max=x_max_left)
        if left_mask
        else None
    )
    contour_right = (
        get_contour(right_mask, x_min=x_min_right, x_max=x_max_right)
        if right_mask
        else None
    )

    ellipse_left = (
        fit_ellipse_to_contour(contour_left) if contour_left is not None else None
    )
    ellipse_right = (
        fit_ellipse_to_contour(contour_right) if contour_right is not None else None
    )

    origin, l_pt, r_pt, min_dist = compute_curve_distance(ellipse_left, ellipse_right)

    width = (top_line[1][0] - top_line[0][0]) / 2 +15
    x_min_left = origin[0] - width
    x_max_left = origin[0]
    x_min_right = origin[0]
    x_max_right = origin[0] + width

    contour_left = (
        get_contour(left_mask, x_min=x_min_left, x_max=x_max_left)
        if left_mask
        else None
    )
    contour_right = (
        get_contour(right_mask, x_min=x_min_right, x_max=x_max_right)
        if right_mask
        else None
    )

    ellipse_left = (
        fit_ellipse_to_contour(contour_left) if contour_left is not None else None
    )
    ellipse_right = (
        fit_ellipse_to_contour(contour_right) if contour_right is not None else None
    )

    if ellipse_left is None or ellipse_right is None:
        print("Skipping frame due to missing ellipses.")
        return None

    y_top = int((top_line[0][1] + top_line[1][1]) / 2)
    y_bot = int((bottom_line[0][1] + bottom_line[1][1]) / 2)

    separation_px = abs(y_top - y_bot)
    separation_m = separation_px * pixel_to_meter
    results["plate_separation_m"] = separation_m

    left_contacts, right_contacts = extract_all_contact_angles(ellipse_left, ellipse_right, y_top, y_bot, top_line, bottom_line)
    # right_contacts = extract_all_contact_angles(ellipse_right, y_top, y_bot, top_line, bottom_line, "right")

    origin, l_pt, r_pt, min_dist = compute_curve_distance(ellipse_left, ellipse_right)

    y0 = origin[1]
    center_l, R2_left = fit_circle_to_contour_near_y(contour_left, y0)
    center_r, R2_right = fit_circle_to_contour_near_y(contour_right, y0)

    results["circle_center_left"] = center_l
    results["circle_radius_R2_left"] = R2_left
    results["circle_center_right"] = center_r
    results["circle_radius_R2_right"] = R2_right

    if origin is None:
        return None

    results["Y*"] = min_dist / 2
    results["origin"] = origin
    results["ellipse_left"] = ellipse_left
    results["ellipse_right"] = ellipse_right
    results["left_contacts"] = left_contacts
    results["right_contacts"] = right_contacts
    results["contour_left"] = contour_left
    results["contour_right"] = contour_right

    plt.figure(figsize=(10, 6))
    for side, ellipse, contacts, contour in [
        ("left", ellipse_left, left_contacts, contour_left),
        ("right", ellipse_right, right_contacts, contour_right),
    ]:
        # Unpack contour into x and y arrays, then stack into (N, 2) shape
        x_rot = contour[:, 0, 0]
        y_rot = contour[:, 0, 1]
        profile_pts_contour = np.stack((x_rot, y_rot), axis=-1)

        

        profile_pts = ellipse_to_points(*ellipse)

        

        meridonal = get_meridonal_profile_new(profile_pts, contacts, side)
        meridonal_contour = get_meridonal_profile_new(
            profile_pts_contour, contacts, side
        )

        plt.plot(meridonal[:, 0], meridonal[:, 1], label=f'Meridonal Profile {side}')
        plt.plot(
            meridonal_contour[:, 0],
            meridonal_contour[:, 1],
            label=f'Meridonal Contour {side}',
            linestyle='--',
        )
        # Interpolate between top_line and bottom_line points
        num_interp = 100
        x_interp_top = np.linspace(top_line[1][0], top_line[0][0], num_interp)
        y_interp_top = np.linspace(top_line[1][1], top_line[0][1], num_interp)
        interp_points_top = np.stack([x_interp_top, y_interp_top], axis=-1)

        x_interp_bottom = np.linspace(bottom_line[1][0], bottom_line[0][0], num_interp)
        y_interp_bottom = np.linspace(bottom_line[1][1], bottom_line[0][1], num_interp)
        interp_points_bottom = np.stack([x_interp_bottom, y_interp_bottom], axis=-1)

        # Plot the substrate lines
        # Use a gradient color for each point along the substrate lines
        colors_top = cm.viridis(np.linspace(0, 1, num_interp))
        colors_bottom = cm.viridis(np.linspace(0, 1, num_interp))
        for i in range(num_interp - 1):
            plt.plot(
            interp_points_top[i : i + 2, 0],
            interp_points_top[i : i + 2, 1],
            color=colors_top[i],
            linewidth=2,
            )
            plt.plot(
            interp_points_bottom[i : i + 2, 0],
            interp_points_bottom[i : i + 2, 1],
            color=colors_bottom[i],
            linewidth=2,
            )
        

        plt.title(f'Meridonal Profile for {side} Side')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()

        def angle_of_substrate(top_line, bottom_line):
            """Calculate angle of substrate line."""
            dx = bottom_line[1][0] - bottom_line[0][0] #bottom right X - bottom left X
            dy = bottom_line[1][1] - bottom_line[0][1]
            bottom_angle = np.degrees(np.arctan2(dy, dx))
            dx = top_line[1][0] - top_line[0][0]
            dy = top_line[1][1] - top_line[0][1]
            top_angle = np.degrees(np.arctan2(dy, dx))
            print(f"Bottom angle: {bottom_angle}, Top angle: {top_angle}")
            return (bottom_angle + top_angle) / 2
        
        Subst_Angle = angle_of_substrate(top_line, bottom_line)
        shifted = transform_points_to_new_frame(
            meridonal[:, 0], meridonal[:, 1], origin, Subst_Angle
        )
        shifted_contour = transform_points_to_new_frame(
            meridonal_contour[:, 0], meridonal_contour[:, 1], origin, Subst_Angle
        )
        R2 = results.get(f"circle_radius_R2_{side}", None)
        print(f"R2 for {side}: {R2}")
        # plt.figure(figsize=(10, 6))
        # plt.plot(shifted[0], shifted[1], label=f'Meridonal Profile {side}')
        # plt.title(f'Meridonal Profile for {side} Side')
        # plt.xlabel('X (shifted)')
        # plt.ylabel('Y (shifted)')
        # plt.legend()

        if shifted[0].size == 0 or shifted[1].size == 0:
            print(f"Skipped H calculation for {side}: empty profile.")
        H_list = []
        H_contour_list = []
        for idx in range(len(shifted[0])):
            kappa = numerical_kappa(shifted, idx)
            H = calculate_mean_curvature(shifted, idx, kappa)
            H2 = new_curvature_calculation(shifted, idx)
        #     H2_contour = new_curvature_calculation(shifted_contour, idx)
            if H2 is not None:
                if side == "left":
                    H2 = -H2
                H_list.append(H2)
        #     if H2_contour is not None:
        #         if side == "left":
        #             H2_contour = -H2_contour
        #         H_contour_list.append(H2_contour)

        # plt.figure(figsize=(10, 6))
        # plt.plot(H_list, label=f'H values for {side} side')
        # plt.plot(H_contour_list, label=f'H contour values for {side} side', linestyle='--')
        # plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        # plt.axvline(0, color='gray', linestyle='--', linewidth=0.5) 
        # plt.title(f'Curvature Profile for {side} Side')
        # plt.xlabel('Index')
        # plt.ylabel('Curvature (H)')
        # plt.legend()
        if H_list:
            H_mean = np.mean(H_list)
        else:
            H_mean = None
            print('H_Mean is None for', side)
        results[f"H_mean_{side}"] = H_mean
        for pos in ["top", "bottom"]:
            contact = contacts[pos]
            if contact["point"] is not None:
                Xn, Yn = transform_point_to_frame(contact["point"], origin)
                if (
                    H_mean is not None
                    and R2 is not None
                    and contact["angle_deg"] is not None
                ):
                    results[f"{side}_{pos}_force"] = calculate_force(
                        H_mean,
                        results["Y*"],
                        abs(contact["angle_deg"]),
                        pixel_to_meter=pixel_to_meter,
                    )
                    results[f"{side}_{pos}_force_circle"] = (
                        calculate_total_force_with_circle_model(
                            contact["point"],
                            origin,
                            contact["angle_deg"],
                            R2,
                            y_star = results["Y*"],
                            pixel_to_meter=pixel_to_meter,
                        )
                    )
                    results[f"{side}_{pos}_angle"] = abs(contact["angle_deg"])
                else:
                    results[f"{side}_{pos}_force"] = None
                    results[f"{side}_{pos}_angle"] = None

                results[f"{side}_{pos}_X"] = Xn
                results[f"{side}_{pos}_Y"] = Yn

    return results, left_mask, right_mask


def visualize_frame_debug(
    img,
    ellipse_left,
    ellipse_right,
    left_contacts,
    right_contacts,
    roi_origin,
    bottom_line,
    top_line,
    mode="plot",
    output_path=None,
    left_mask=None,
    right_mask=None,
    result=None,
    origin = None,
):

    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    img_debug = img.copy()

    # Optional: Draw left/right masks
    mask_overlay = img_debug.copy()
    for mask, color in zip(
        [left_mask, right_mask], [(0, 255, 0), (255, 0, 0)]
    ):  # green for left, red for right
        if mask is not None:
            binary = np.array(mask["segmentation"]).astype(np.uint8)
            mask_overlay[binary > 0] = (
                mask_overlay[binary > 0] * 0.7 + np.array(color) * 0.3
            ).astype(np.uint8)
    img_debug = mask_overlay

    # Draw substrate lines
    cv2.line(img_debug, bottom_line[0], bottom_line[1], (255, 255, 0), 2)
    cv2.line(img_debug, top_line[0], top_line[1], (255, 0, 0), 2)

    # Draw fitted ellipses
    for ellipse, color in [(ellipse_left, (0, 0, 255)), (ellipse_right, (0, 255, 0))]:
        if ellipse:
            (xc, yc), (MA, ma), angle = ellipse
            cv2.ellipse(
                img_debug,
                (int(xc), int(yc)),
                (int(MA // 2), int(ma // 2)),
                angle,
                0,
                360,
                color,
                2,
            )

    for center, radius, color in [
        (
            result.get("circle_center_left"),
            result.get("circle_radius_R2_left"),
            (255, 100, 255),
        ),
        (
            result.get("circle_center_right"),
            result.get("circle_radius_R2_right"),
            (100, 255, 255),
        ),
    ]:
        if center is not None and radius is not None:
            x, y = map(int, center)
            r = int(radius)
            cv2.circle(img_debug, (x, y), r, color, 2)

    if origin is not None:
        origin = to_int(origin)
        cv2.circle(img_debug, origin, 6, (255, 0, 255), -1) # Draw origin point
    # Draw contact angles
    img_debug = draw_contact_angle_debug(
        img_debug, left_contacts["top"], roi_origin, (0, 0, 255), 0, "L-T: "
    )
    img_debug = draw_contact_angle_debug(
        img_debug, left_contacts["bottom"], roi_origin, (0, 0, 255), 0, "L-B: "
    )
    img_debug = draw_contact_angle_debug(
        img_debug, right_contacts["top"], roi_origin, (0, 255, 0), 0, "R-T: "
    )
    img_debug = draw_contact_angle_debug(
        img_debug, right_contacts["bottom"], roi_origin, (0, 255, 0), 0, "R-B: "
    )

    # Draw origin marker
    origin_shifted = (int(roi_origin[0]), int(roi_origin[1]))
    cv2.circle(img_debug, origin_shifted, 6, (255, 0, 255), -1)

    # Show result
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_debug, cv2.COLOR_BGR2RGB))
    plt.title("Ellipses, Contact Points, and Substrate Lines")
    plt.axis("off")
    if mode == "plot":
        plt.show()
    elif mode == "save":
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))
            # print(f"Debug image saved to {output_path}")
        else:
            print("Output path not specified. Debug image not saved.")


def process_single_frame(args):
    idx, image_path, mask_dir, bottom_lines, top_lines, pixel_to_meter, debug_dir = args
    json_path = mask_dir / f"{image_path.stem}_masks.json"
    if not json_path.exists():
        print(f"Mask file {json_path} does not exist for frame {idx}. Skipping.")
        return None

    try:
        with open(json_path, "r") as f:
            masks = json.load(f)

        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        top_line = top_lines[idx]
        bottom_line = bottom_lines[idx]

        result, left_mask, right_mask = process_frame_with_ellipses(
            img, masks, bottom_line, top_line
        )

        # Optional: Save debug frame
        if result and debug_dir:
            debug_out = Path(debug_dir) / f"debug_{image_path.stem}.png"
            visualize_frame_debug(
                img=img,
                ellipse_left=result["ellipse_left"],
                ellipse_right=result["ellipse_right"],
                left_contacts=result["left_contacts"],
                right_contacts=result["right_contacts"],
                roi_origin=(0, 0),
                bottom_line=bottom_line,
                top_line=top_line,
                mode="save",
                output_path=str(debug_out),
                left_mask=left_mask,
                right_mask=right_mask,
                result=result,
                origin=result.get("origin", None),
            )
            print(f"Debug image saved to {debug_out}")

        return result

    except Exception as e:
        print(f"Failed on frame {idx}: {e}")
        return None


if __name__ == "__main__":
    # %% Example use on first frame
    image_paths = sorted(image_dir.glob("*.tif")) or sorted(image_dir.glob("*.png"))
    frame_number = 62  # Change this to process a different frame
    json_path = mask_dir / f"{image_paths[frame_number].stem}_masks.json"
    print(json_path)
    with open(json_path, "r") as f:
        masks = json.load(f)
    img = cv2.imread(str(image_paths[frame_number]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bottom_line = bottom_lines[frame_number]
    top_line = top_lines[frame_number]

    frame_result, left_mask, right_mask = process_frame_with_ellipses(
        img, masks, bottom_line, top_line
    )
    plt.show()
    if frame_result:
        print("Left Top Force (Ellipse Model):", frame_result["left_top_force"])
        print("Right Top Force (Ellipse Model):", frame_result["right_top_force"])
        print("Left Bottom Force (Ellipse Model):", frame_result["left_bottom_force"])
        print("Right Bottom Force (Ellipse Model):", frame_result["right_bottom_force"])
        print("Left Top Force (Circle Model):", frame_result["left_top_force_circle"])
        print("Right Top Force (Circle Model):", frame_result["right_top_force_circle"])
        print("Left Bottom Force (Circle Model):", frame_result["left_bottom_force_circle"])
        print("Right Bottom Force (Circle Model):", frame_result["right_bottom_force_circle"])
    else:
        print("Failed to process frame.")

    debug_img = img.copy()
    h, w = img.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        debug_img = cv2.resize(
            debug_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    cv2.polylines(
        debug_img,
        [frame_result["contour_left"]],
        isClosed=False,
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.polylines(
        debug_img,
        [frame_result["contour_right"]],
        isClosed=False,
        color=(0, 255, 0),
        thickness=2,
    )
    mask_overlay = debug_img.copy()
    for mask, color in zip(
        [left_mask, right_mask], [(0, 255, 0), (255, 0, 0)]
    ):  # green for left, red for right
        if mask is not None:
            binary = np.array(mask["segmentation"]).astype(np.uint8)
            mask_overlay[binary > 0] = (
                mask_overlay[binary > 0] * 0.7 + np.array(color) * 0.3
            ).astype(np.uint8)
    debug_img = mask_overlay
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title("Contours Used for Ellipse Fitting")
    plt.axis("off")
    plt.show()

    if frame_result:
        visualize_frame_debug(
            img=img,
            ellipse_left=frame_result["ellipse_left"],
            ellipse_right=frame_result["ellipse_right"],
            left_contacts=frame_result["left_contacts"],
            right_contacts=frame_result["right_contacts"],
            roi_origin=(0, 0),
            bottom_line=bottom_line,
            top_line=top_line,
            mode="plot",
            origin = frame_result.get("origin"),
            # output_path=str(debug_out),  # Uncomment to save debug image
            left_mask=left_mask,
            right_mask=right_mask,
            result=frame_result,
        )
