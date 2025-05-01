import cv2
import numpy as np
import os
import csv
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog

# Define left and right catenary functions
def left_catenary(y, a, y0, c):
    """Catenary curve for left edge (opens to the right)"""
    return -a * np.cosh((y - y0) / a) + c

def right_catenary(y, a, y0, c):
    """Catenary curve for right edge (opens to the left)"""
    return a * np.cosh((y - y0) / a) + c

def left_ellipse(y, h, k, a, b):
    """Safer ellipse calculation"""
    with np.errstate(invalid='ignore'):
        return h + a * np.sqrt(1 - ((y-k)/b)**2)

def right_ellipse(y, h, k, a, b):
    """Safer right-opening ellipse"""
    with np.errstate(invalid='ignore'):
        return h - a * np.sqrt(1 - ((y-k)/b)**2)

def select_roi(image):
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    return roi

def residual_filter(x_data, y_data, model_func, initial_params, n_std=2):
    """Simple residual-based outlier filtering"""
    try:
        # Initial fit to identify outliers
        params, _ = curve_fit(model_func, y_data, x_data, p0=initial_params, maxfev=2000)
        
        # Calculate residuals
        predicted = model_func(y_data, *params)
        residuals = x_data - predicted
        
        # Filter points within n_std standard deviations
        residual_std = np.std(residuals)
        mask = np.abs(residuals) <= n_std * residual_std
        
        return x_data[mask], y_data[mask]
    except:
        # Fallback if initial fit fails
        return x_data, y_data

def calculate_contact_angle(curve, y_pos, rotation_matrix):
    """Calculate angle for both catenary and ellipse"""
    if curve['type'] == 'catenary':
        a, y0, c = curve['coeffs']
        dx_dy = np.sinh((y_pos - y0)/a) 
        dx_dy *= -1 if curve['func'] == left_catenary else 1
        
    elif curve['type'] == 'ellipse':
        h, k, a, b = curve['coeffs']
        term = ((y_pos - k)/b)**2
        sqrt_term = np.sqrt(np.clip(1 - term, 1e-6, 1))
        dx_dy = (a/(b**2)) * (y_pos - k)/sqrt_term
        dx_dy *= -1 if curve['func'] == left_ellipse else 1
        
    direction_vec = np.array([1, dx_dy])
    rotated_vec = np.dot(rotation_matrix, direction_vec)
    angle = np.arctan2(rotated_vec[1], rotated_vec[0])
    return np.degrees(angle) % 180

import numpy as np

def ellipse_parametric_points(xc, yc, a, b, angle_deg, num_points=1000):
    theta = np.deg2rad(angle_deg)
    t = np.linspace(0, 2*np.pi, num_points)
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    x = xc + a * cos_t * np.cos(theta) - b * sin_t * np.sin(theta)
    y = yc + a * cos_t * np.sin(theta) + b * sin_t * np.cos(theta)
    return x, y, t

def find_top_bottom_intersections(x, y, t, roi_top, roi_bottom):
    # Get index closest to top and bottom y-values
    idx_top = np.argmin(np.abs(y - roi_top))
    idx_bottom = np.argmin(np.abs(y - roi_bottom))
    return (x[idx_top], y[idx_top], t[idx_top]), (x[idx_bottom], y[idx_bottom], t[idx_bottom])

def ellipse_tangent_angle(a, b, t_val, angle_deg):
    # Derivatives of x(t) and y(t)
    theta = np.deg2rad(angle_deg)
    dx_dt = -a * np.sin(t_val) * np.cos(theta) - b * np.cos(t_val) * np.sin(theta)
    dy_dt = -a * np.sin(t_val) * np.sin(theta) + b * np.cos(t_val) * np.cos(theta)
    angle_rad = np.arctan2(dy_dt, dx_dt)
    return np.rad2deg(angle_rad) % 180
def compute_cv2_ellipse_rmse(x_data, y_data, xc, yc, major, minor, angle_deg, num_samples=500):
    # Parametric ellipse points
    a, b = major / 2, minor / 2
    theta = np.deg2rad(angle_deg)
    t = np.linspace(0, 2*np.pi, num_samples)
    cos_t, sin_t = np.cos(t), np.sin(t)

    x_ellipse = xc + a * cos_t * np.cos(theta) - b * sin_t * np.sin(theta)
    y_ellipse = yc + a * cos_t * np.sin(theta) + b * sin_t * np.cos(theta)

    ellipse_points = np.column_stack((x_ellipse, y_ellipse))
    data_points = np.column_stack((x_data, y_data))

    # Find nearest ellipse point for each data point
    diffs = data_points[:, np.newaxis, :] - ellipse_points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # Shape (num_data_points, num_ellipse_points)
    min_dists = np.min(dists, axis=1)

    rmse = np.sqrt(np.mean(min_dists**2))
    return rmse



def get_initial_guesses(x_data, y_data, side, roi_x, roi_width):
    y_center = np.mean(y_data)
    y_range = np.ptp(y_data)  # Peak-to-peak range
    
    # Catenary parameters (unchanged)
    a_cat = roi_width * 0.1
    c_cat = roi_x + (0.2 * roi_width if side == "left" else 0.8 * roi_width)
    
    # Improved ellipse parameters
    h_ell = roi_x + (0.35 * roi_width if side == "left" else 0.65 * roi_width)
    k_ell = y_center
    a_ell = 0.2 * roi_width  # Semi-major axis
    b_ell = max(0.5 * y_range, 10)  # Ensure minimum minor axis length
    
    return {
        'catenary': [a_cat, y_center, c_cat],
        'ellipse': [h_ell, k_ell, a_ell, b_ell]
    }

def get_bounds(side, roi_x, roi_width, model_type):
    """More flexible physics-based constraints"""
    if model_type == 'catenary':
        return (
            [0.05*roi_width, -np.inf, roi_x + (0 if side == 'left' else 0.5*roi_width)],
            [3*roi_width, np.inf, roi_x + (0.5*roi_width if side == 'left' else roi_width)]
        )
    elif model_type == 'ellipse':
        return (
            [roi_x - 0.3*roi_width,  # More flexible x-center
             -np.inf, 
             0.05*roi_width,  # Minimum major axis
             5],  # Minimum minor axis
            [roi_x + 1.3*roi_width, 
             np.inf, 
             roi_width, 
             3*roi_width]
        )

def iterative_fit(x_data, y_data, model_func, initial_params, bounds, max_iter=5):
    """Robust fitting with parameter validation"""
    # Ensure initial parameters are within bounds
    params = np.clip(initial_params, bounds[0], bounds[1])
    best_params = params.copy()
    best_rmse = np.inf
    
    for i in range(max_iter):
        try:
            params, _ = curve_fit(
                model_func, y_data, x_data,
                p0=best_params,
                bounds=bounds,
                method='trf',
                max_nfev=2000*(i+1)
            )
            
            # Validate parameters stay within bounds
            params = np.clip(params, bounds[0], bounds[1])
            
            residuals = x_data - model_func(y_data, *params)
            current_rmse = np.sqrt(np.mean(residuals**2))
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_params = params
                print(f"Iteration {i+1}: RMSE improved to {best_rmse:.2f}")

        except Exception as e:
            print(f"Iteration {i+1} failed: {str(e)}")
            continue
            
    return best_params, best_rmse

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    roi = select_roi(image)
    if roi == (0, 0, 0, 0):
        return
    
    x_roi, y_roi, w, h = roi
    cv2.rectangle(output, (x_roi, y_roi), (x_roi+w, y_roi+h), (0, 255, 0), 2)
    
    # Edge detection
    roi_img = image[y_roi:y_roi+h, x_roi:x_roi+w]
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 100)
    
    edge_points = np.column_stack(np.where(edges > 0))
    if len(edge_points) < 10:
        print("Not enough edge points found")
        return
    
    abs_points = edge_points + [y_roi, x_roi]
    abs_points = abs_points[:, [1, 0]]  # (x, y)
    
    # Split into left/right regions
    left_mask = abs_points[:, 0] < x_roi + w*0.35
    right_mask = abs_points[:, 0] > x_roi + w*0.65
    left_points, right_points = abs_points[left_mask], abs_points[right_mask]
    
    curves = []
    output_data = []
    for points, color, side in [(left_points, (0, 255, 0), 'left'), 
                                (right_points, (0, 0, 255), 'right')]:
        if len(points) < 50:
            print(f"Skipping {side} side - only {len(points)} points")
            continue
            
        xs, ys = points[:, 0], points[:, 1]
        print(f"\nProcessing {side} side with {len(xs)} points")
        # # Plot xs, ys points on the output image in pink
        # for x, y in zip(xs, ys):
        #     cv2.circle(output, (int(x), int(y)), 2, (255, 105, 180), -1)  # Pink color (BGR: 180, 105, 255)
        guesses = get_initial_guesses(xs, ys, side, x_roi, w)
        if not guesses:
            continue
            
        models = []
        
        for model_type in ['catenary', 'ellipse']:
            print(f"\n--- {model_type.upper()} FIT ATTEMPT ---")
            func = left_catenary if (side == 'left' and model_type == 'catenary') else \
                   right_catenary if (side == 'right' and model_type == 'catenary') else \
                   left_ellipse if (side == 'left' and model_type == 'ellipse') else \
                   right_ellipse

            # Apply residual-based filtering
            x_filt, y_filt = residual_filter(xs, ys, func, guesses[model_type])
            for x, y in zip(x_filt, y_filt):
                cv2.circle(output, (int(x), int(y)), 2, (180, 105, 255), -1)
            pts_for_cv2 = np.column_stack((x_filt, y_filt)).astype(np.int32)
            print(f"After residual filtering: {len(x_filt)}/{len(xs)} points remain")
            
            if len(x_filt) < 20:
                print("Insufficient points for fitting")
                continue
            if model_type == 'ellipse':
                if len(pts_for_cv2) >= 5:  # cv2.fitEllipse requires at least 5 points
                    ellipse = cv2.fitEllipse(pts_for_cv2)
                    (xc, yc), (major, minor), angle = ellipse
                    print(f"OpenCV Ellipse Fit: center=({xc:.1f}, {yc:.1f}), axes=({major:.1f}, {minor:.1f}), angle={angle:.1f}")
                    rmse = compute_cv2_ellipse_rmse(x_filt, y_filt, xc, yc, major, minor, angle)
                    models.append({
                        'type': 'cv2_ellipse',
                        'func': None,
                        'coeffs': (xc, yc, major, minor, angle),
                        'rmse': rmse,
                        'points': (x_filt, y_filt)
                    })
                    print(f"RMSE (cv2 ellipse): {rmse:.2f} pixels")

            if model_type == 'catenary':
                # Get adaptive bounds
                bounds = get_bounds(side, x_roi, w, model_type)
                print(f"Using bounds:\nLower: {bounds[0]}\nUpper: {bounds[1]}")
                
                try:
                    params, rmse = iterative_fit(x_filt, y_filt, func,
                                                guesses[model_type], bounds)
                    print(f"Successful {model_type} fit!")
                    print(f"Parameters: {params}")
                    print(f"RMSE: {rmse:.2f} pixels")
                    
                    models.append({
                        'type': model_type,
                        'func': func,
                        'coeffs': params,
                        'rmse': rmse,
                        'points': (x_filt, y_filt)
                    })
                    
                except RuntimeError as e:
                    print(f"{model_type} fitting failed: {str(e)}")
                    continue
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    continue

        if models:
            best_model = sorted(models, key=lambda x: x['rmse'])[0]
            print(f"\nBest model: {best_model['type']} with RMSE {best_model['rmse']:.2f}")
            for model in models:

                if model['type'] == 'cv2_ellipse':
                    y_min, y_max = np.min(y_filt), np.max(y_filt)
                        
                    (xc, yc, major, minor, angle) = model['coeffs']
                    a, b = major / 2, minor / 2

                    x_vals, y_vals, t_vals = ellipse_parametric_points(xc, yc, a, b, angle)
                    roi_top = y_roi
                    roi_bottom = y_roi + h

                    (top_x, top_y, t_top), (bot_x, bot_y, t_bot) = find_top_bottom_intersections(x_vals, y_vals, t_vals, roi_top, roi_bottom)

                    top_angle = ellipse_tangent_angle(a, b, t_top, angle)
                    bottom_angle = ellipse_tangent_angle(a, b, t_bot, angle)

                    # Store angles and points in output_data
                    output_data.append({
                        'side': side,
                        'model': model,
                        'top_angle': top_angle,
                        'bottom_angle': bottom_angle,
                        'top_point': (top_x, top_y),
                        'bottom_point': (bot_x, bot_y)
                    })
                # Calculate angles with actual rotation matrix
                if model['type'] == 'catenary':
                    # continue # Skip angle calculation for catenaries
                    try:
                        rotation_matrix = np.eye(2)  # Replace with actual calculation
                        y_min, y_max = np.min(y_filt), np.max(y_filt)
                        
                        top_angle = calculate_contact_angle(model, y_min, rotation_matrix)
                        bottom_angle = calculate_contact_angle(model, y_max, rotation_matrix)
                        
                        output_data.append({
                            'side': side,
                            'model': model,
                            'top_angle': top_angle,
                            'bottom_angle': bottom_angle,
                            'top_point': (model['func'](y_min, *model['coeffs']), y_min),
                            'bottom_point': (model['func'](y_max, *model['coeffs']), y_max)
                        })
                    except Exception as e:
                        print(f"Angle calculation failed: {str(e)}")
        for result in output_data:
            side = result['side']
            model = result['model']
            color = (0, 255, 0) if side == 'left' else (0, 0, 255)
            
            # Generate points for plotting
            y_min, y_max = np.min(result['model']['points'][1]), np.max(result['model']['points'][1])
            y_vals = np.linspace(y_min, y_max, 100)
            
            if model['type'] == 'cv2_ellipse':
                (xc, yc, major, minor, angle) = model['coeffs']
                ellipse_box = ((xc, yc), (major, minor), angle)
                cv2.ellipse(output, ellipse_box, color, 2)

                cv2.putText(output, "Press any key to continue...", (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Results", output)
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed

            if model['type'] == 'catenary':
                try:
                    # Calculate curve points
                    x_vals = model['func'](y_vals, *model['coeffs'])
                    
                    # Create points array and draw
                    points = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2).astype(np.int32)
                    cv2.polylines(output, [points], False, color, 2)
                    
                    # Draw angle markers
                    cv2.circle(output, (int(result['top_point'][0]), int(result['top_point'][1])), 5, (255, 0, 0), -1)
                    cv2.circle(output, (int(result['bottom_point'][0]), int(result['bottom_point'][1])), 5, (0, 255, 255), -1)
                    # Wait for user to press any key to continue 
                    cv2.putText(output, "Press any key to continue...", (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow("Results", output)
                    cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                except Exception as e:
                    print(f"Error drawing {model['type']} curve: {str(e)}")

        # Add legend
        cv2.putText(output, "Catenary", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, "Ellipse", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
        

    # Save results (same as before)
    output_dir = os.path.join(os.path.dirname(image_path), "analyzed_results")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    with open(os.path.join(output_dir, f"{base_name}_analysis.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "Side", "Model", "RMSE",
            "Param1", "Param2", "Param3", "Param4",
            "TopAngle", "BottomAngle",
            "TopX", "TopY", "BottomX", "BottomY"
        ])
        
        for data in output_data:
            model = data.get('model')
            if model is None:
                continue
            
            coeffs = model.get('coeffs', [])
            if len(coeffs) < 4:
                coeffs = list(coeffs) + ['N/A'] * (4 - len(coeffs))
            
            writer.writerow([
                os.path.basename(image_path),
                data["side"],
                model['type'],
                f"{model['rmse']:.4f}",
                *coeffs,
                f"{data.get('top_angle', 'N/A')}",
                f"{data.get('bottom_angle', 'N/A')}",
                *data['top_point'],
                *data['bottom_point']
            ])
    
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_processed.png"), output)
    cv2.imshow("Results", output)
    cv2.waitKey(3000)   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.attributes('-topmost', True)  # Keep the file dialog on top
    image_dir = filedialog.askdirectory(title="Select Image Directory")
    if not image_dir:
        print("No directory selected. Exiting.")
        exit(1)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing: {filename}")
            process_image(image_path)