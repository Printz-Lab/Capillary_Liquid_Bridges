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

def select_roi(image):
    print("Select ROI and press ENTER or SPACE. Press ESC to cancel.")
    roi = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = roi
    return x, y, x + w, y + h  # (x_min, y_min, x_max, y_max)

def draw_detected_substrate_roi(image, roi_coords):
    x_min, y_min, x_max, y_max = roi_coords
    img = image.copy()
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Auto-detected ROI from Substrates")
    plt.axis('off')
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

def process_droplet_two_lobes(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return

    # ROI selection
    roi_coords = select_roi(image)
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
    edges_left = preprocess_image(left_img)
    contour_left = find_largest_contour(edges_left)
    ellipse_left = fit_ellipse_to_contour(contour_left) if contour_left is not None else None

    # Process right side
    edges_right = preprocess_image(right_img)
    contour_right = find_largest_contour(edges_right)
    ellipse_right = fit_ellipse_to_contour(contour_right) if contour_right is not None else None

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

    # draw_results_on_full_image(image, (x_min, y_min, x_max, y_max), contour_left, contour_right, ellipse_left, ellipse_right)
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

def contact_angle_at_index(points, index, side='left', label='top'):
    if index <= 0 or index >= len(points) - 1:
        return None
    p1 = points[index - 1]
    p2 = points[index + 1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = np.arctan2(np.abs(dy), np.abs(dx))
    angle_deg = np.rad2deg(angle_rad)
    # print(f"Contact angle at index {index} ({label}): {angle_deg:.2f} degrees")
    # Flip convention for left side
    if side == 'left' and label == 'top' or side == 'right' and label == 'bottom':
        angle_deg = -angle_deg
    return angle_deg
def find_contact_point_on_line_half(points, line_y, side='right', tolerance=5):
    # Get center x to split
    center_x = np.mean(points[:, 0])
    if side == 'right':
        relevant_points = points[points[:, 0] > center_x]
    else:
        relevant_points = points[points[:, 0] < center_x]

    dists = np.abs(relevant_points[:, 1] - line_y)
    close_indices = np.where(dists < tolerance)[0]
    if len(close_indices) == 0:
        return None
    best_idx = close_indices[np.argmin(dists[close_indices])]
    return best_idx, relevant_points[best_idx]

def extract_all_contact_angles(ellipse, roi_y_top, roi_y_bottom, side='left'):
    if ellipse is None:
        return {}

    center, axes, angle = ellipse
    points = ellipse_to_points(center, axes, angle)

    result = {}

    for label, line_y in [('top', roi_y_top), ('bottom', roi_y_bottom)]:
        side_selector = 'right' if side == 'left' else 'left'  # inward-facing side
        idx, pt = find_contact_point_on_line_half(points, line_y, side_selector)
        if pt is not None:
            ang = contact_angle_at_index(points, idx, side, label)
            result[label] = {'point': pt, 'angle_deg': ang}
        else:
            result[label] = {'point': None, 'angle_deg': None}

    return result
def draw_debug_overlay(full_image, roi_coords, ellipse_left=None, ellipse_right=None, 
                        contacts_left=None, contacts_right=None, origin=None):
    x_min, y_min, x_max, y_max = roi_coords
    img = full_image.copy()

    # Draw ROI boundary (optional)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1) #color: yellow

    # Draw substrate lines
    cv2.line(img, (x_min, y_min), (x_max, y_min), (255, 0, 255), 1)  # Top line, color: magenta
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
    for contacts, color in [(contacts_left, (0, 0, 255)), (contacts_right, (0, 255, 0))]:
        if contacts:
            for pos in ['top', 'bottom']:
                pt = contacts[pos]['point']
                if pt is not None:
                    shifted_pt = (int(pt[0] + x_min), int(pt[1] + y_min))
                    cv2.circle(img, shifted_pt, 5, color, -1)
    if origin is not None:
            # Shift origin from cropped coordinates to full image coordinates
            shifted_origin = (int(origin[0] + x_min), int(origin[1] + y_min))
            
            # Vertical axis (X_new) - cyan line
            cv2.line(img, (shifted_origin[0], y_min), (shifted_origin[0], y_max), (255, 255, 0), 2)
            
            # Horizontal axis (Y_new) - cyan line
            cv2.line(img, (x_min, shifted_origin[1]), (x_max, shifted_origin[1]), (255, 255, 0), 2)

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Ellipses, Contact Points, and New Axes")
    plt.axis('off')
    plt.show()

def compute_curve_distance(ellipse_left, ellipse_right, num_samples=500):
    # Generate points for both ellipses
    pts_l = ellipse_to_points(*ellipse_left)
    pts_r = ellipse_to_points(*ellipse_right)
    
    # Find pair of closest points between the curves
    min_dist = float('inf')
    closest_pair = None
    
    # Brute-force search for closest points (optimize this if needed)
    for pl in pts_l:
        for pr in pts_r:
            dx = pl[0] - pr[0]
            dy = pl[1] - pr[1]
            dist = dx*dx + dy*dy  # Squared distance for efficiency
            if dist < min_dist:
                min_dist = dist
                closest_pair = (pl, pr)
    
    if closest_pair is None:
        return None, None, None, None
    
    # Set origin as midpoint between closest points
    left_pt, right_pt = closest_pair
    origin = (
        (left_pt[0] + right_pt[0])/2, 
        (left_pt[1] + right_pt[1])/2
    )
    
    return origin, left_pt, right_pt, np.sqrt(min_dist)

def transform_points_to_new_frame(xs, ys, origin):
    ox, oy = origin
    # new X = (y - oy)   (vertical displ)
    # new Y = (x - ox)   (lateral displ)
    X_new = ys - oy
    Y_new = xs - ox
    return X_new, Y_new

def transform_point_to_frame(pt, origin):
    """
    pt: (x_pixel, y_pixel) in cropped image coords
    origin: (x0, y0) in cropped image coords
    returns: (X_new, Y_new) = (vertical, lateral) relative to origin
    """
    x0, y0 = origin
    x_pt, y_pt = pt
    Xn = y_pt - y0    # up/down from origin → new X
    Yn = x_pt - x0    # left/right from origin → new Y
    return Xn, Yn

def shift_point_to_full_image(pt, roi_origin, half_width_offset=0):
    x_shift = roi_origin[0] + half_width_offset
    y_shift = roi_origin[1]
    return np.array([pt[0] + x_shift, pt[1] + y_shift])

def draw_contact_angle_debug(full_image, contact, roi_origin, color, half_width_offset=0, label=""):
    img = full_image.copy()

    pt = contact['point']
    angle = contact['angle_deg']
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
    cv2.putText(img, f"{label}{angle:.1f}°", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
    denominator = (e + np.cos(u)) * np.sqrt(e**2 - np.cos(u)**2 + epsilon)
    return np.cos(u) / denominator

def nodoid_param(t, a, b, num_points=100):
    """Safe parameterization with validation"""
    try:
        # Parameter validation
        if a <= 0 or b <= 0:
            raise ValueError(f"Invalid parameters a={a}, b={b}")
            
        e = np.sqrt(a**2 + b**2)/a
        if e <= 1:
            raise ValueError(f"Invalid nodoid eccentricity e={e:.2f}")

        y = b * np.sqrt((e - np.cos(t)) / (e + np.cos(t)))
        x = np.zeros_like(t)
        
        # Vectorized integration
        for i, ti in enumerate(t):
            if ti < -np.pi/2 or ti > np.pi/2:
                raise ValueError(f"Parameter t={ti:.2f} out of range")
            x[i], _ = quad(integrand, 0, ti, args=(e,))
            
        x = (b**2/a) * x
        return x, y
        
    except Exception as e:
        print(f"Nodoid parameterization failed: {str(e)}")
        return np.array([]), np.array([])  # Return empty arrays

def unduloid_param(t, a, b, num_points=100):
    e = np.sqrt(a**2 - b**2) / a
    y = b * np.sqrt((1 - e * np.cos(t)) / (1 + e * np.cos(t)))
    # Precompute integral values for efficiency
    t_vals = np.linspace(0, t, num_points)
    integrand_vals = 1 / ((1 + e * np.cos(t_vals)) * np.sqrt(1 - (e**2) * np.cos(t_vals)**2))
    x = (b**2 / a) * np.trapz(integrand_vals, t_vals)
    return x, y

def calculate_parameters(bridge_type, y, yc1, yc2, theta1, theta2):
    if bridge_type == "nodoid":
        a = 0.5 * ( (yc1**2 - y**2)/(y - yc1*np.sin(theta1)) )
        b_sq = y * yc1 * (yc1 - y*np.sin(theta1)) / (y - yc1*np.sin(theta1))
        return a, np.sqrt(b_sq)
    elif bridge_type == "unduloid":
        a = 0.5 * ( (yc1**2 - y**2)/(yc1*np.sin(theta1) - y) )
        b_sq = y * yc1 * (yc1 - y*np.sin(theta1)) / (yc1*np.sin(theta1) - y)
        return a, np.sqrt(b_sq)
    else:
        return None, None
    
def transform_to_image_coords(x_param, y_param, origin, roi_coords):
    """With validation checks"""
    x_shift = origin[0] + roi_coords[0]
    y_shift = origin[1] + roi_coords[1]
    
    # Convert to numpy arrays for vectorization
    x_img = np.asarray(x_param) + x_shift
    y_img = y_shift - np.asarray(y_param)
    
    # Check bounds
    if (np.min(x_img) < 0) or (np.max(x_img) > roi_coords[2]):
        print(f"X coordinates out of bounds: {np.min(x_img):.1f}-{np.max(x_img):.1f}")
    if (np.min(y_img) < 0) or (np.max(y_img) > roi_coords[3]):
        print(f"Y coordinates out of bounds: {np.min(y_img):.1f}-{np.max(y_img):.1f}")
    
    return x_img, y_img

def draw_theoretical_curve(ax, bridge_type, a, b, origin, roi_coords):
    """Safe plotting with empty data checks"""
    if bridge_type not in ["nodoid", "unduloid"]:
        return None
    
    try:
        t = np.linspace(-np.pi/2, np.pi/2, 100)
        
        if bridge_type == "nodoid":
            x, y = nodoid_param(t, a, b)
        else:
            x, y = unduloid_param(t, a, b)

        # Critical check for valid data
        if len(x) == 0 or len(y) == 0:
            print(f"No valid {bridge_type} points to plot")
            return None

        x_img, y_img = transform_to_image_coords(x, y, origin, roi_coords)
        
        # Safe line creation
        lines = ax.plot(x_img, y_img, 'r--', linewidth=2, 
                       label=f'Theoretical {bridge_type}')
        
        return lines[0] if lines else None
    
    except Exception as e:
        print(f"Plotting error: {str(e)}")
        return None




if __name__ == "__main__":
    # Hide the root Tk window
    root = tk.Tk()
    root.withdraw()

    # Ask user to pick a folder
    folder_path = filedialog.askdirectory(title="Select folder containing droplet images")
    if not folder_path:
        print("No folder selected, exiting.")
        exit()

    # Gather image files
    exts = (".jpg", ".jpeg", ".png", ".tif", ".bmp")
    image_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith(exts)]

    if not image_files:
        print("No image files found.")
        exit()

    results = []

    # Process images
    for image_path in image_files:
        print(f"\nProcessing {os.path.basename(image_path)}…")
        result = process_droplet_two_lobes(image_path)
        if not result:
            continue

        # Unpack results
        ellipse_left, ellipse_right, (x_min, y_min, x_max, y_max), image, cropped = result
        roi_height, roi_width = y_max-y_min, x_max-x_min

        # Contact angle analysis
        left_contacts = extract_all_contact_angles(ellipse_left, 0, cropped.shape[0]-1, 'left')
        right_contacts = extract_all_contact_angles(ellipse_right, 0, cropped.shape[0]-1, 'right')

        # Geometric analysis
        origin, l_pt, r_pt, min_dist = compute_curve_distance(ellipse_left, ellipse_right)
        if not origin:
            print("Skipping distance calculations")
            continue

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
                    "y": None
                }
        Ystar = min_dist/2 if min_dist else None
        data["y"] = Ystar

        # Process contact points and angles
        def process_contacts(contacts, side, origin):
            for pos in ['top', 'bottom']:
                contact = contacts.get(pos, {})
                if contact['point'] is not None and contact['angle_deg'] is not None:
                    # Transform coordinates
                    Xn, Yn = transform_point_to_frame(contact['point'], origin)
                    # Store values with absolute angle
                    data[f"{side}_{pos}_angle"] = abs(contact['angle_deg'])
                    data[f"{side}_{pos}_Y"] = Yn
                    data[f"{side}_{pos}_X"] = Xn

        if origin:
            process_contacts(left_contacts, 'left', origin)
            process_contacts(right_contacts, 'right', origin)

        results.append(data)
        

        # Transform coordinates
        pts_l = ellipse_to_points(*ellipse_left)
        pts_r = ellipse_to_points(*ellipse_right)
        Xl, Yl = transform_points_to_new_frame(pts_l[:,0], pts_l[:,1], origin)
        Xr, Yr = transform_points_to_new_frame(pts_r[:,0], pts_r[:,1], origin)

        # Generate debug visualization
        debug_img = image.copy()
        
        # Draw fitted ellipses on full image
        if ellipse_left is not None:
            (lx, ly), (lma, lmi), lang = ellipse_left
            cv2.ellipse(debug_img, 
                        (int(lx+x_min), int(ly+y_min)),
                        (int(lma/2), int(lmi/2)),
                        lang, 0, 360, (0,0,255), 2)

        if ellipse_right is not None:
            (rx, ry), (rma, rmi), rang = ellipse_right
            cv2.ellipse(debug_img, 
                        (int(rx+x_min), int(ry+y_min)),
                        (int(rma/2), int(rmi/2)),
                        rang, 0, 360, (255,0,0), 2)

        # Draw coordinate axes
        origin_full = (int(origin[0]+x_min), int(origin[1]+y_min))
        cv2.line(debug_img, (origin_full[0], y_min), (origin_full[0], y_max), (0,255,255), 2)
        cv2.line(debug_img, (x_min, origin_full[1]), (x_max, origin_full[1]), (0,255,255), 2)
        
        # Draw contact points and angles
        debug_img = draw_contact_angle_debug(debug_img, left_contacts['top'], (x_min,y_min), (255,0,0), 0, "L-T: ")
        debug_img = draw_contact_angle_debug(debug_img, left_contacts['bottom'], (x_min,y_min), (255,0,0), 0, "L-B: ")
        debug_img = draw_contact_angle_debug(debug_img, right_contacts['top'], (x_min,y_min), (0,255,0), 0, "R-T: ")
        debug_img = draw_contact_angle_debug(debug_img, right_contacts['bottom'], (x_min,y_min), (0,255,0), 0, "R-B: ")

        # Show results
        plt.figure(figsize=(12,8))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Analysis Results - Fitted Ellipses and Contact Points")
        plt.axis('off')
        plt.show()

        # Comprehensive printout
        print(f"\n=== Results for {os.path.basename(image_path)} ===")
        #transform contact points for below

    
        # Contact points and angles
        print("\nIntersection Points and Angles:")
        for side, contacts in [("Left", left_contacts), ("Right", right_contacts)]:
            for pos in ["Top", "Bottom"]:
                contact = contacts[pos.lower()]
                pt = contact['point']
                angle = contact['angle_deg']
                
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
        y = data['y']
        side = 'right'  # Try with left side first
        yc1 = data[f'{side}_top_Y']
        yc2 = data[f'{side}_bottom_Y']
        theta1 = np.deg2rad(abs(data[f'{side}_top_angle']))  # Absolute value for angle direction
        theta2 = np.deg2rad(abs(data[f'{side}_bottom_angle']))

        # After contact angle calculation
        bridge_type = classify_bridge(y, yc1, yc2, np.radians(theta1), np.radians(theta2))
        a, b = calculate_parameters(bridge_type, y, yc1, yc2, np.radians(theta1), np.radians(theta2))

      
        # Get experimental points for error calculation
        pts_experimental = np.vstack([pts_l, pts_r])

        # Enhanced visualization
        # In your main processing loop:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Store all plottable artists
        # In your main processing loop:
        # In your plotting section:
        artists = []
        theory_line = draw_theoretical_curve(ax, bridge_type, a, b, origin, (x_min, y_min))

        if theory_line:  # Only add if valid
            artists.append(theory_line)

        # Add experimental points
        exp_points = ax.scatter(pts_experimental[:,0], pts_experimental[:,1], 
                            c='blue', s=10, label='Experimental')
        artists.append(exp_points)

        # Create legend only if we have entries
        if artists:
            ax.legend(handles=artists)
        else:
            ax.text(0.5, 0.5, 'No Valid Theoretical Curve', 
                ha='center', va='center', transform=ax.transAxes)

        plt.show()
        # After parameter calculation
        print(f"Bridge type: {bridge_type}")
        print(f"Parameters: a={a:.2f}, b={b:.2f}")
        print(f"ROI coords: x={x_min}-{x_max}, y={y_min}-{y_max}")
        print(f"Origin point: {origin_full}")