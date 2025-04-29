import cv2
import numpy as np
import os
import csv
from scipy.optimize import curve_fit

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
        return h - a * np.sqrt(np.clip(1 - ((y-k)/b)**2, 1e-6, 1))

def right_ellipse(y, h, k, a, b):
    """Safer right-opening ellipse"""
    with np.errstate(invalid='ignore'):
        return h + a * np.sqrt(np.clip(1 - ((y-k)/b)**2, 1e-6, 1))

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

def get_initial_guesses(x_data, y_data, side, roi_x, roi_width):
    y_center = np.mean(y_data)
    y_range = np.max(y_data) - np.min(y_data)
    
    # Catenary parameters (unchanged)
    a_cat = roi_width * 0.1
    c_cat = roi_x + (0.2 * roi_width if side == "left" else 0.8 * roi_width)
    
    # Improved ellipse parameters
    h_ell = roi_x + (0.4 * roi_width if side == "left" else 0.6 * roi_width)
    k_ell = y_center
    a_ell = 0.15 * roi_width  # Reduced initial major axis
    b_ell = 0.4 * y_range     # More conservative minor axis
    
    return {
        'catenary': [a_cat, y_center, c_cat],
        'ellipse': [h_ell, k_ell, a_ell, b_ell]
    }

def get_bounds(side, roi_x, roi_width, model_type):
    """Improved physics-based constraints"""
    if model_type == 'catenary':
        return (
            [0.1*roi_width, -np.inf, roi_x + (0 if side == 'left' else 0.6*roi_width)],
            [2*roi_width, np.inf, roi_x + (0.4*roi_width if side == 'left' else roi_width)]
        )
    elif model_type == 'ellipse':
        return (
            [roi_x + (0 if side == 'left' else 0.5*roi_width), 
             -np.inf, 0.1*roi_width, 0.1*roi_width],
            [roi_x + (0.5*roi_width if side == 'left' else roi_width), 
             np.inf, roi_width, 2*roi_width]
        )

def iterative_fit(x_data, y_data, model_func, initial_params, bounds, max_iter=5):
    """Robust iterative fitting with adaptive learning"""
    best_params = initial_params.copy()
    best_rmse = np.inf
    success = False  # Track if any iteration succeeded
    
    for i in range(max_iter):
        try:
            # Check parameter feasibility
            if not np.all(np.logical_and(
                np.array(bounds[0]) <= best_params,
                best_params <= np.array(bounds[1])
            )):
                print(f"Iteration {i+1}: Initial params out of bounds")
                continue
                
            # Perform curve fitting
            params, _ = curve_fit(
                model_func, y_data, x_data,
                p0=best_params,
                bounds=bounds,
                method='trf',
                max_nfev=2000*(i+1),
                ftol=1e-4,
                xtol=1e-4
            )
            
            # Calculate RMSE
            residuals = x_data - model_func(y_data, *params)
            current_rmse = np.sqrt(np.mean(residuals**2))
            
            if current_rmse < best_rmse:
                best_rmse = current_rmse
                best_params = params
                success = True
                print(f"Iteration {i+1}: RMSE improved to {best_rmse:.2f}")

                # Adjust bounds conservatively
                bounds = (
                    np.clip(params*0.9, [b*0.95 for b in bounds[0]], [b*1.05 for b in bounds[1]]),
                    np.clip(params*1.1, [b*0.95 for b in bounds[0]], [b*1.05 for b in bounds[1]])
                )
                
        except Exception as e:
            print(f"Iteration {i+1} failed: {str(e)}")
            continue
            
    if not success:
        raise RuntimeError("All iterations failed to converge")
        
    return best_params, best_rmse

def show_fit_debug(image, x_data, y_data, fits, roi):
    """Fixed coordinate system visualization"""
    x_roi, y_roi, w, h = roi
    debug_img = cv2.cvtColor(image[y_roi:y_roi+h, x_roi:x_roi+w], cv2.COLOR_GRAY2BGR)
    
    # Convert absolute coordinates to ROI-relative
    rel_x = x_data - x_roi
    rel_y = y_data - y_roi
    
    for x, y in zip(rel_x, rel_y):
        cv2.circle(debug_img, (int(x), int(y)), 2, (0,0,255), -1)
        
    # Plot fits in ROI coordinates
    y_range = np.linspace(0, h, 100)
    colors = [(0,255,0), (255,0,0)]
    
    for fit, color in zip(fits, colors):
        try:
            x_vals = fit['func'](y_range + y_roi, *fit['coeffs']) - x_roi
            pts = np.column_stack([x_vals, y_range]).astype(int)
            cv2.polylines(debug_img, [pts], False, color, 2)
        except:
            continue
            
    cv2.imshow("Fitting Debug", debug_img)
    cv2.waitKey(100)

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
            print(f"After residual filtering: {len(x_filt)}/{len(xs)} points remain")
            
            if len(x_filt) < 20:
                print("Insufficient points for fitting")
                continue
                
            # Get adaptive bounds
            bounds = get_bounds(side, x_roi, w, model_type)
            print(f"Using bounds:\nLower: {bounds[0]}\nUpper: {bounds[1]}")
            
            # In the model fitting loop:
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
            
            # Calculate angles with actual rotation matrix
            try:
                rotation_matrix = np.eye(2)  # Replace with actual calculation
                y_min, y_max = np.min(y_filt), np.max(y_filt)
                
                top_angle = calculate_contact_angle(best_model, y_min, rotation_matrix)
                bottom_angle = calculate_contact_angle(best_model, y_max, rotation_matrix)
                
                output_data.append({
                    'side': side,
                    'model': best_model,
                    'top_angle': top_angle,
                    'bottom_angle': bottom_angle,
                    'top_point': (best_model['func'](y_min, *best_model['coeffs']), y_min),
                    'bottom_point': (best_model['func'](y_max, *best_model['coeffs']), y_max)
                })
            except Exception as e:
                print(f"Angle calculation failed: {str(e)}")

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
    image_dir = r"C:\Users\ezrap\OneDrive\Documents\Spring 2025 HW\Printz Lab Research\Capillary Bridging\Anton's snapshots\APTMS Filtered"
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing: {filename}")
            process_image(image_path)