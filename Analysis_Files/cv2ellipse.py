import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.lines as mlines
from scipy.spatial import distance



def select_roi(image):
    print("Select ROI and press ENTER or SPACE. Press ESC to cancel.")
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = roi
    return x, y, x + w, y + h  # (x_min, y_min, x_max, y_max)


def auto_detect_substrate_roi(
    image, hough_thresh=50, min_line_len=100, margin_frac=0.55
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")
    plt.show()
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=20,
    )
    h, w = image.shape[:2]
    top_lines = []
    bottom_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # horizontal line
                y_avg = (y1 + y2) // 2
                if y_avg < h * margin_frac:
                    top_lines.append(y_avg)
                elif y_avg > h * (1 - margin_frac):
                    bottom_lines.append(y_avg)

    if not top_lines or not bottom_lines:
        print("Failed to detect substrate lines.")
        return None
    top_lines = [max(top_lines)]
    bottom_lines = [min(bottom_lines)]
    print("Top lines:", top_lines)
    print("Bottom lines:", bottom_lines)
    y_min = top_lines[0]
    y_max = bottom_lines[0]
    x_min = 5
    x_max = w - 5

    return (x_min, y_min, x_max, y_max)


def draw_detected_substrate_roi(image, roi_coords):
    x_min, y_min, x_max, y_max = roi_coords
    img = image.copy()
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Auto-detected ROI from Substrates")
    plt.axis("off")
    plt.show()


def preprocess_image(image, blur_ksize=5, canny_thresh=(50, 150)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(blurred, *canny_thresh)
    return edges


def find_largest_contour(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def split_contour_top_bottom(contour, buffer_px=5):
    contour = contour.reshape(-1, 2)
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])
    y_center = (y_min + y_max) / 2

    top = contour[contour[:, 1] < (y_center - buffer_px)]
    bottom = contour[contour[:, 1] > (y_center + buffer_px)]

    return top.reshape(-1, 1, 2), bottom.reshape(-1, 1, 2)


def split_contour_left_right(contour, buffer_px=5):
    contour = contour.reshape(-1, 2)
    print("Sample contour points (first 5):", contour[:5])  # debug line

    # Just to be explicit:
    xs = contour[:, 0]
    ys = contour[:, 1]

    x_min = np.min(xs)
    x_max = np.max(xs)
    x_center = (x_min + x_max) / 2

    # Filter based on x-coordinates
    left = contour[xs < (x_center - buffer_px)]
    right = contour[xs > (x_center + buffer_px)]

    print(f"Left points: {len(left)}, Right points: {len(right)}")  # debug line

    return left.reshape(-1, 1, 2), right.reshape(-1, 1, 2)


def fit_ellipse_to_contour(contour):
    return cv2.fitEllipse(contour) if len(contour) >= 5 else None


def crop_left_right(image):
    h, w = image.shape[:2]
    mid = w // 2
    left = image[:, :mid]
    right = image[:, mid:]
    return left, right


def draw_results_on_full_image(
    full_image, roi_coords, contour_left, contour_right, ellipse_left, ellipse_right
):
    x_min, y_min, x_max, y_max = roi_coords
    overlay = full_image.copy()

    # Shift contours and ellipses into full image coordinates
    if contour_left is not None:
        shifted_left = contour_left + np.array([[[x_min, y_min]]])
        cv2.drawContours(overlay, [shifted_left], -1, (0, 255, 0), 1)

    if ellipse_left:
        (xc, yc), (MA, ma), angle = ellipse_left
        shifted_ellipse_left = ((xc + x_min, yc + y_min), (MA, ma), angle)
        cv2.ellipse(overlay, shifted_ellipse_left, (255, 0, 0), 2)

    if contour_right is not None:
        shifted_right = contour_right + np.array([[[x_min, y_min]]])
        cv2.drawContours(overlay, [shifted_right], -1, (0, 255, 0), 1)

    if ellipse_right:
        (xc, yc), (MA, ma), angle = ellipse_right
        shifted_ellipse_right = ((xc + x_min, yc + y_min), (MA, ma), angle)
        cv2.ellipse(overlay, shifted_ellipse_right, (0, 0, 255), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Fitted Ellipses on Full Image")
    plt.axis("off")
    plt.show()


def process_droplet_two_lobes(image_path, roi_type="manual", canny_thresh=(50, 150)):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # ROI selection
    if roi_type == "manual":
        roi_coords = select_roi(image)
    elif roi_type == "auto":
        # Auto-detect ROI based on substrate
        roi_coords = auto_detect_substrate_roi(image)

    if roi_coords is None:
        return
    x_min, y_min, x_max, y_max = roi_coords
    cropped = image[y_min:y_max, x_min:x_max]

    draw_detected_substrate_roi(image, roi_coords)

    # From your ROI selection
    top_y = y_min
    bottom_y = y_max

    # Split ROI into left and right
    left_img, right_img = crop_left_right(cropped)

    # Process left side
    edges_left = preprocess_image(left_img, canny_thresh=canny_thresh)
    contour_left = find_largest_contour(edges_left)
    ellipse_left = (
        fit_ellipse_to_contour(contour_left) if contour_left is not None else None
    )

    # Process right side
    edges_right = preprocess_image(right_img, canny_thresh=canny_thresh)
    contour_right = find_largest_contour(edges_right)
    ellipse_right = (
        fit_ellipse_to_contour(contour_right) if contour_right is not None else None
    )
    
    
    # Offset right ellipse by half-width to align on original image
    h, w = cropped.shape[:2]
    if ellipse_right:
        (xc, yc), (MA, ma), angle = ellipse_right
        ellipse_right = ((xc + w // 2, yc), (MA, ma), angle)
        contour_right = contour_right + np.array([[[w // 2, 0]]])

    # Combine for visualization
    all_contours = []
    if contour_left is not None:
        all_contours.append(contour_left)
    if contour_right is not None:
        all_contours.append(contour_right)

    draw_results_on_full_image(
        image,
        (x_min, y_min, x_max, y_max),
        contour_left,
        contour_right,
        ellipse_left,
        ellipse_right,
    )
    return ellipse_left, ellipse_right, (x_min, y_min, x_max, y_max), image, cropped


def ellipse_to_points(center, axes, angle_deg, num_points=360):
    cx, cy = center
    a, b = axes[0] / 2, axes[1] / 2  # major and minor radii
    theta = np.deg2rad(angle_deg)

    t = np.linspace(0, 2 * np.pi, num_points)
    cos_angle = np.cos(theta)
    sin_angle = np.sin(theta)

    x = a * np.cos(t)
    y = b * np.sin(t)

    x_rot = cos_angle * x - sin_angle * y + cx
    y_rot = sin_angle * x + cos_angle * y + cy

    return np.stack((x_rot, y_rot), axis=-1)  # shape: (N, 2)


def contact_angle_at_index(points, index, side="left", label="top"):
    if index <= 0 or index >= len(points) - 1:
        return None
    # Average angle over a window of points around the index
    window = 3  # Use 3 points before and after
    start = max(0, index - window)
    end = min(len(points) - 1, index + window)
    dxs = []
    dys = []
    for i in range(start, end):
        p1 = points[i]
        p2 = points[i + 1]
        dxs.append(p2[0] - p1[0])
        dys.append(p2[1] - p1[1])
    dx = np.mean(dxs)
    dy = np.mean(dys)
    angle_rad = np.arctan2(np.abs(dy), np.abs(dx))
    angle_deg = np.rad2deg(angle_rad)
    # Flip convention for left side
    if side == "left" and label == "top" or side == "right" and label == "bottom":
        angle_deg = -angle_deg
    # print(f"Contact angle at index {index} ({label}): {angle_deg:.2f} degrees")
    return angle_deg


def find_contact_point_on_line_half(points, line_y, top_line, bottom_line, label, side="right", tolerance=20):
    # center_x = (top_line[0][0] + top_line[1][0]) / 2
    center_x = points[:, 0].mean()
    if side == "right":
        relevant_points = points[points[:, 0] < center_x]
    else:
        relevant_points = points[points[:, 0] > center_x]

    dists = np.abs(relevant_points[:, 1] - line_y)
    close_indices = np.where(dists < tolerance)[0]
    if len(close_indices) == 0:
        print(f"No contact point found on {label} line for {side} side within tolerance.")
        return None
    best_idx = close_indices[np.argmin(dists[close_indices])]
    return best_idx, relevant_points[best_idx], relevant_points

def extract_all_contact_angles(ellipse_left, ellipse_right, roi_y_top, roi_y_bottom, top_line, bottom_line):
    if ellipse_left is None or ellipse_right is None:
        return {}, {}

    plt.figure()
    points_left = ellipse_to_points(ellipse_left[0], ellipse_left[1], ellipse_left[2])
    points_right = ellipse_to_points(ellipse_right[0], ellipse_right[1], ellipse_right[2])
    plt.plot(points_left[:, 0], points_left[:, 1], 'o', markersize=2, label='Left Ellipse Points')
    plt.plot(points_right[:, 0], points_right[:, 1], 'o', markersize=2, label='Right Ellipse Points')
    plt.scatter(top_line[1][0], top_line[1][1], color='r', label='Top Line Y right')
    plt.scatter(bottom_line[1][0], bottom_line[1][1], color='g', label='Bottom Line Y right')
    plt.scatter(top_line[0][0], top_line[0][1], color='b', label='Top Line Y left')
    plt.scatter(bottom_line[0][0], bottom_line[0][1], color='y', label='Bottom Line Y left')
    plt.axvline(points_left[:, 0].mean(), color='b', linestyle='--', label='Left Ellipse Mean Y')
    plt.axvline(points_right[:, 0].mean(), color='r', linestyle='--', label='Right Ellipse Mean Y')
    plt.axhline(y=roi_y_top, color='r', linestyle='--', label='ROI Top Line')
    plt.axhline(y=roi_y_bottom, color='g', linestyle='--', label='ROI Bottom Line')



    left_result = {}
    right_result = {}

    if ellipse_left is not None:
        center, axes, angle = ellipse_left
        points = ellipse_to_points(center, axes, angle)

        for label, line_y in [("top", roi_y_top), ("bottom", roi_y_bottom)]:
            res = find_contact_point_on_line_half(points, line_y, top_line, bottom_line, label, side="left")
            if res is not None:
                idx, pt, relevant_points = res
                plt.plot(pt[0], pt[1], 'ro', markersize=5, label=f'Left {label} Contact Point')
                ang = contact_angle_at_index(relevant_points, idx, "left", label)
                if ang is not None:
                    dx = np.cos(np.radians(-ang))
                    dy = np.sin(np.radians(-ang))
                    pt2 = pt + np.array([dx, dy]) * 10  # Offset point
                    pt1 = pt - np.array([dx, dy]) * 10
                    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g--', label=f'Left {label} Angle Line')
                left_result[label] = {"point": pt, "angle_deg": ang}
            else:
                left_result[label] = {"point": None, "angle_deg": None}

    if ellipse_right is not None:
        center, axes, angle = ellipse_right
        points = ellipse_to_points(center, axes, angle)
        for label, line_y in [("top", roi_y_top), ("bottom", roi_y_bottom)]:
            res = find_contact_point_on_line_half(points, line_y, top_line, bottom_line, label, side="right")
            if res is not None:
                idx, pt, relevant_points = res
                plt.plot(pt[0], pt[1], 'go', markersize=5, label=f'Right {label} Contact Point')
                ang = contact_angle_at_index(relevant_points, idx, "right", label)
                if ang is not None:
                    dx = np.cos(np.radians(-ang))
                    dy = np.sin(np.radians(-ang))
                    pt2 = pt + np.array([dx, dy]) * 10  # Offset point
                    pt1 = pt - np.array([dx, dy]) * 10
                    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'g--', label=f'Right {label} Angle Line')
                right_result[label] = {"point": pt, "angle_deg": ang}
            else:
                right_result[label] = {"point": None, "angle_deg": None}
    plt.legend()
    # plt.show()
    plt.close()
    return left_result, right_result

def get_meridonal_profile_new(points, contacts, side):
    """
    Extract the meridional profile points from an ellipse fit.
    """
    top_contact = contacts["top"]['point']
    bottom_contact = contacts["bottom"]['point']

    # Skip if either point is missing
    if top_contact is None or bottom_contact is None:
        return np.empty((0, 2))  # return empty array to skip this profile

    if side == 'left':
        mask = (
            (points[:, 0] > min(top_contact[0], bottom_contact[0])) &
            (points[:, 1] > top_contact[1]) &
            (points[:, 1] < bottom_contact[1])
        )
    elif side == 'right':
        mask = (
            (points[:, 0] < max(top_contact[0], bottom_contact[0])) &
            (points[:, 1] > top_contact[1]) &
            (points[:, 1] < bottom_contact[1])
        )
    else:
        raise ValueError("side must be 'left' or 'right'")

    return points[mask]

def get_meriodonal_profile(points, contacts, side):
    points = np.array(points)
    if side == "left":
        # Left side: use the leftmost points
        top_contact = contacts["top"]['point']
        bottom_contact = contacts['bottom']['point']
        profile_points = points[
            (points[:, 0] <= max(top_contact[0], bottom_contact[0])) &
             (points[:, 1] <= top_contact[1]) & 
            (points[:, 1] >= bottom_contact[1])
             ]
    if side == 'right':
        # Right side: use the rightmost points
        top_contact = contacts["top"]['point']
        bottom_contact = contacts['bottom']['point']
        profile_points = points[
            (points[:, 0] >= min(top_contact[0], bottom_contact[0])) &
            (points[:, 1] <= top_contact[1]) & 
            (points[:, 1] >= bottom_contact[1])
            ]
    
    return profile_points

def draw_meriodonal_profile(
    full_image, roi_coords, profile_points, side="left", color=(255, 0, 0)
):
    x_min, y_min, x_max, y_max = roi_coords
    img = full_image.copy()

    # Shift points to full image coordinates
    shifted_points = profile_points + np.array([x_min, y_min])

    # Draw the meriodonal profile
    for pt in shifted_points:
        cv2.circle(img, tuple(pt.astype(int)), 2, color, -1)

    # Draw ROI boundary (optional)
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1
    )  # color: yellow

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Meriodonal Profile on {side} Side")
    plt.axis("off")
    plt.show()

# Utility to fit circle from y-cropped contour
def fit_circle_to_contour_near_y(contour, y0, y_margin=50):
    pts = contour.reshape(-1, 2)
    subset = pts[np.abs(pts[:, 1] - y0) < y_margin]
    if len(subset) >= 3:
        A = np.c_[2 * subset[:, 0], 2 * subset[:, 1], np.ones(len(subset))]
        b = subset[:, 0]**2 + subset[:, 1]**2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc, c = sol
        radius = np.sqrt(c + xc**2 + yc**2)
        return (xc, yc), radius
    return None, None


def draw_debug_overlay(
    full_image,
    roi_coords,
    ellipse_left=None,
    ellipse_right=None,
    contacts_left=None,
    contacts_right=None,
    origin=None,
):
    x_min, y_min, x_max, y_max = roi_coords
    img = full_image.copy()

    # Draw ROI boundary (optional)
    cv2.rectangle(
        img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1
    )  # color: yellow

    # Draw substrate lines
    cv2.line(
        img, (x_min, y_min), (x_max, y_min), (255, 0, 255), 1
    )  # Top line, color: magenta
    cv2.line(img, (x_min, y_max), (x_max, y_max), (255, 0, 255), 1)  # Bottom line

    # Draw ellipses
    if ellipse_left:
        (xc, yc), (MA, ma), angle = ellipse_left
        shifted = ((xc + x_min, yc + y_min), (MA, ma), angle)
        cv2.ellipse(img, shifted, (0, 0, 255), 2)
    if ellipse_right:
        (xc, yc), (MA, ma), angle = ellipse_right
        shifted = ((xc + x_min, yc + y_min), (MA, ma), angle)
        cv2.ellipse(img, shifted, (0, 255, 0), 2)

    # Draw contact points
    for contacts, color in [
        (contacts_left, (0, 0, 255)),
        (contacts_right, (0, 255, 0)),
    ]:
        if contacts:
            for pos in ["top", "bottom"]:
                pt = contacts[pos]["point"]
                if pt is not None:
                    shifted_pt = (int(pt[0] + x_min), int(pt[1] + y_min))
                    cv2.circle(img, shifted_pt, 5, color, -1)
    if origin is not None:
        # Shift origin from cropped coordinates to full image coordinates
        shifted_origin = (int(origin[0] + x_min), int(origin[1] + y_min))

        # Vertical axis (X_new) - cyan line
        cv2.line(
            img,
            (shifted_origin[0], y_min),
            (shifted_origin[0], y_max),
            (255, 255, 0),
            2,
        )

        # Horizontal axis (Y_new) - cyan line
        cv2.line(
            img,
            (x_min, shifted_origin[1]),
            (x_max, shifted_origin[1]),
            (255, 255, 0),
            2,
        )

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Ellipses, Contact Points, and New Axes")
    plt.axis("off")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def fit_poly_and_get_angle(contour, y_contact, side, degree=4, window=200, debug_image=None, label="top"):
    """
    Fit polynomial to a contour segment near y_contact, return tangent angle and contact point.
    Optionally overlay on image for debugging.
    """
    if contour is None or len(contour) < degree + 1:
        return None, None

    contour = contour.reshape(-1, 2)
    distances = np.abs(contour[:, 1] - y_contact)
    sorted_indices = np.argsort(distances)
    nearest_points = contour[sorted_indices[:window]]

    y = nearest_points[:, 0]
    x = nearest_points[:, 1]

    if np.ptp(x) < 5:  # prevent poor fits
        return None, None

    # Fit polynomial y(x)
    p = Polynomial.fit(x, y, degree).convert()
    dp = p.deriv()

    x_fit = np.linspace(min(x)-100, max(x)+100, 500)
    y_fit = p(x_fit)

    # Compute angle from slope at the x of closest point
    distances = np.abs(x_fit - y_contact)
    idx_min = np.argmin(distances)
    x_contact = y_fit[idx_min]
    xx_contact = x_fit[idx_min]
    dxdy = dp(xx_contact)
    dydx= 1 / dxdy if dxdy != 0 else np.inf  # Avoid division by zero
    angle_rad = np.arctan(np.abs(dydx))
    angle_deg = np.degrees(angle_rad)

    if side == "left" and label == "top" or side == "right" and label == "bottom":
        angle_deg = -angle_deg

    # Optional debug plot
    if debug_image is not None:
        
        
        plt.plot(contour[:, 0], contour[:, 1], 'g.', alpha=0.2, label=f"{side} contour", markersize=1)
        plt.plot(y_fit, x_fit, 'r-', label=f"{side} poly fit")
        plt.scatter([x_contact], [y_contact], c='cyan', label=f"{side} contact point")
        plt.title(f"{side.capitalize()} {label} Polynomial Fit & Contact Angle: {angle_deg:.1f}°")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.tight_layout()

    contact_point = np.array([x_contact, y_contact])
    return angle_deg, contact_point, p


def compute_curve_distance(contour_left, contour_right, top_k=15):
    contour_left = contour_left.reshape(-1, 2)
    contour_right = contour_right.reshape(-1, 2)

    # Compute full distance matrix between all points
    dists = distance.cdist(contour_left, contour_right)  # shape (N, M)
    pairs = []

    # Find the indices of the top_k smallest distances
    flat_indices = np.argpartition(dists.flatten(), top_k)[:top_k]
    rows, cols = np.unravel_index(flat_indices, dists.shape)

    for i in range(len(rows)):
        pt_l = contour_left[rows[i]]
        pt_r = contour_right[cols[i]]
        pairs.append((pt_l, pt_r))

    # Average midpoints of top_k closest pairs
    midpoints = np.array([(pl + pr) / 2 for pl, pr in pairs])
    origin = np.mean(midpoints, axis=0)

    # Also return avg of left/right points (optional)
    left_pt = np.mean([pl for pl, _ in pairs], axis=0)
    right_pt = np.mean([pr for _, pr in pairs], axis=0)
    mean_dist = np.mean([np.linalg.norm(pl - pr) for pl, pr in pairs])

    # # Debug plot
    # plt.figure(figsize=(10, 6))
    # plt.plot(contour_left[:, 0], contour_left[:, 1], 'b.', markersize=1, label='Left Contour')
    # plt.plot(contour_right[:, 0], contour_right[:, 1], 'r.', markersize=1, label='Right Contour')
    # plt.plot(origin[0], origin[1], 'go', markersize=6, label='Averaged Origin')
    # for pl, pr in pairs:
    #     plt.plot([pl[0], pr[0]], [pl[1], pr[1]], 'k-', alpha=0.2)  # connecting lines
    # plt.title(f"Average of {top_k} Closest Point Pairs")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    return origin, left_pt, right_pt, mean_dist


def transform_points_to_new_frame(xs, ys, origin, angle_deg=0):
    ox, oy = origin
    # new X = (y - oy)   (vertical displ)
    # new Y = (x - ox)   (lateral displ)
    X_new = ys - oy
    Y_new = xs - ox

    # Apply rotation if angle is specified
    if angle_deg != 0:
        angle_rad = np.deg2rad(angle_deg)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        X_new_rotated = X_new * cos_angle - Y_new * sin_angle
        Y_new_rotated = X_new * sin_angle + Y_new * cos_angle
        X_new, Y_new = X_new_rotated, Y_new_rotated
    return X_new, Y_new


def transform_point_to_frame(pt, origin):
    """
    pt: (x_pixel, y_pixel) in cropped image coords
    origin: (x0, y0) in cropped image coords
    returns: (X_new, Y_new) = (vertical, lateral) relative to origin
    """
    x0, y0 = origin
    x_pt, y_pt = pt
    Xn = y_pt - y0  # up/down from origin → new X
    Yn = x_pt - x0  # left/right from origin → new Y
    return Xn, Yn


def shift_point_to_full_image(pt, roi_origin, half_width_offset=0):
    x_shift = roi_origin[0] + half_width_offset
    y_shift = roi_origin[1]
    return np.array([pt[0] + x_shift, pt[1] + y_shift])


def draw_contact_angle_debug(
    full_image, contact, roi_origin, color, half_width_offset=0, label=""
):
    img = full_image.copy()

    pt = contact["point"]
    angle = contact["angle_deg"]
    if pt is None or angle is None:
        return img

    # Shift contact point to full image
    pt_full = shift_point_to_full_image(pt, roi_origin, half_width_offset)
    x0, y0 = int(pt_full[0]), int(pt_full[1])

    # Draw point
    cv2.circle(img, (x0, y0), 4, color, -1)

    # Compute tangent vector
    length = 40  # length of tangent line

    angle_rad = np.deg2rad(angle)
    dx = int(length * np.cos(angle_rad))
    dy = -int(length * np.sin(angle_rad))

    # Tangent line
    pt1 = (x0 - dx, y0 - dy)
    pt2 = (x0 + dx, y0 + dy)
    cv2.line(img, pt1, pt2, color, 2)

    # Annotate angle
    text_pos = (x0 + 5, y0 - 10)
    cv2.putText(
        img, f"{label}{angle:.1f} deg", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
    )

    return img


################################## - NODOID - ############################################################################################################################################
def classify_bridge(y, yc1, yc2, theta1, theta2):
    yc1_sin = yc1 * np.sin(theta1)
    yc2_sin = yc2 * np.sin(theta2)

    if max(yc1_sin, yc2_sin) < y < min(yc1, yc2):
        return "nodoid"
    elif 0 < y < min(yc1_sin, yc2_sin):
        return "unduloid"
    elif y == yc1_sin or y == yc2_sin:
        return "catenoid"
    else:
        return "unknown"


def integrand(u, e, epsilon=1e-8):
    denominator = (e + np.cos(u)) * np.sqrt(e**2 - np.cos(u) ** 2 + epsilon)
    return np.cos(u) / denominator


def nodoid_param(t, a, b, num_points=100):
    """Safe parameterization with validation"""
    try:
        # Parameter validation
        if a <= 0 or b <= 0:
            raise ValueError(f"Invalid parameters a={a}, b={b}")

        e = np.sqrt(a**2 + b**2) / a
        if e <= 1:
            raise ValueError(f"Invalid nodoid eccentricity e={e:.2f}")

        y = b * np.sqrt((e - np.cos(t)) / (e + np.cos(t)))
        x = np.zeros_like(t)

        # Vectorized integration
        for i, ti in enumerate(t):
            if ti < -np.pi / 2 or ti > np.pi / 2:
                raise ValueError(f"Parameter t={ti:.2f} out of range")
            x[i], _ = quad(integrand, 0, ti, args=(e,))

        x = (b**2 / a) * x
        return x, y

    except Exception as e:
        print(f"Nodoid parameterization failed: {str(e)}")
        return np.array([]), np.array([])  # Return empty arrays


def unduloid_param(t, a, b, num_points=100):
    e = np.sqrt(a**2 - b**2) / a
    y = b * np.sqrt((1 - e * np.cos(t)) / (1 + e * np.cos(t)))
    # Precompute integral values for efficiency
    t_vals = np.linspace(0, t, num_points)
    integrand_vals = 1 / (
        (1 + e * np.cos(t_vals)) * np.sqrt(1 - (e**2) * np.cos(t_vals) ** 2)
    )
    x = (b**2 / a) * np.trapz(integrand_vals, t_vals)
    return x, y


def calculate_parameters(bridge_type, y, yc1, yc2, theta1, theta2):
    if bridge_type == "nodoid":
        a = 0.5 * ((yc1**2 - y**2) / (y - yc1 * np.sin(theta1)))
        b_sq = y * yc1 * (yc1 - y * np.sin(theta1)) / (y - yc1 * np.sin(theta1))
        return a, np.sqrt(b_sq)
    elif bridge_type == "unduloid":
        a = 0.5 * ((yc1**2 - y**2) / (yc1 * np.sin(theta1) - y))
        b_sq = y * yc1 * (yc1 - y * np.sin(theta1)) / (yc1 * np.sin(theta1) - y)
        return a, np.sqrt(b_sq)
    else:
        return None, None


def transform_to_image_coords(x_param, y_param, origin, roi_coords, side):
    x_shift = origin[0] + roi_coords[0]
    y_shift = origin[1] + roi_coords[1]
    if side == "right":
        # Convert to numpy arrays for vectorization
        x_img = np.asarray(x_param) + y_shift
        y_img = np.asarray(y_param) + x_shift

    elif side == "left":
        # Convert to numpy arrays for vectorization
        x_img = np.asarray(x_param) + y_shift
        y_img = -np.asarray(y_param) + x_shift
    return x_img, y_img


def draw_theoretical_curve(ax, bridge_type, a, b, origin, roi_coords, side):
    """Safe plotting with empty data checks"""
    if bridge_type not in ["nodoid", "unduloid"]:
        return None

    try:
        t = np.linspace(-np.pi / 2, np.pi / 2, 100)

        if bridge_type == "nodoid":
            x, y = nodoid_param(t, a, b)
        else:
            x, y = unduloid_param(t, a, b)

        # Critical check for valid data
        if len(x) == 0 or len(y) == 0:
            print(f"No valid {bridge_type} points to plot")
            return None

        x_img, y_img = transform_to_image_coords(x, y, origin, roi_coords, side)

        return x_img, y_img

    except Exception as e:
        print(f"{str(e)}")
        return None


if __name__ == "__main__":
    # Hide the root Tk window
    print("Run the other file!")