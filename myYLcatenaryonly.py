import cv2
import numpy as np
import os
import csv
from scipy.optimize import curve_fit
#fixed something
# Define left and right catenary functions
def left_catenary(y, a, y0, c):
    """Catenary curve for left edge (opens to the right)"""
    return -a * np.cosh((y - y0) / a) + c

def right_catenary(y, a, y0, c):
    """Catenary curve for right edge (opens to the left)"""
    return a * np.cosh((y - y0) / a) + c

def left_ellipse(y, h, k, a, b):
    """Left-opening ellipse (x = h - a*sqrt(1 - ((y-k)/b)^2)"""
    return h - a * np.sqrt(np.clip(1 - ((y - k)/b)**2, 1e-6, 1))

def right_ellipse(y, h, k, a, b):
    """Right-opening ellipse (x = h + a*sqrt(1 - ((y-k)/b)^2)"""
    return h + a * np.sqrt(np.clip(1 - ((y - k)/b)**2, 1e-6, 1))

def select_roi(image):
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    return roi

def model_outlier_filter(x_data, y_data, model_func, p0, bounds):
    """Generalized outlier filtering for any model"""
    try:
        coeffs, _ = curve_fit(
            model_func,
            y_data, 
            x_data,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        predicted = model_func(y_data, *coeffs)
        residuals = np.abs(x_data - predicted)
        threshold = np.median(residuals) + .5 * np.std(residuals)
        return (
            x_data[residuals < threshold], 
            y_data[residuals < threshold],
            coeffs,
            residuals
        )
    except Exception as e:
        print(f"Filtering failed: {str(e)}")
        return x_data, y_data, None, None

def find_waist(y_values, left_curve, right_curve):
    """Find waist location between two catenaries"""
    x_left = left_curve['func'](y_values, *left_curve['coeffs'])
    x_right = right_curve['func'](y_values, *right_curve['coeffs'])
    distances = np.abs(x_right - x_left)
    return y_values[np.argmin(distances)]

def transform_coordinates(x, y, origin, rotation_matrix):
    """Convert image coordinates to rotated system"""
    centered = np.array([x - origin[0], y - origin[1]])
    rotated = np.dot(rotation_matrix, centered)
    return rotated[0], rotated[1]

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

def fit_models(xs, ys, side, roi_x, roi_width):
    """Fit both catenary and ellipse models with outlier filtering"""
    models = []
    
    # Catenary fitting
    if side == "left":
        model_func = left_catenary
        c_lower = roi_x
        c_upper = roi_x + 0.35 * roi_width
    else:
        model_func = right_catenary
        c_lower = roi_x + 0.65 * roi_width
        c_upper = roi_x + roi_width
        
    a_guess = (np.max(xs) - np.min(xs)) / 4
    c_guess = np.clip(np.median(xs), c_lower, c_upper)
    p0_cat = [a_guess, np.median(ys), c_guess]
    bounds_cat = ([1e-3, -np.inf, c_lower], [np.inf, np.inf, c_upper])
    
    x_cat, y_cat, coeffs_cat, res_cat = model_outlier_filter(
        xs, ys, model_func, p0_cat, bounds_cat
    )
    
    if coeffs_cat is not None:
        models.append({
            'type': 'catenary',
            'func': model_func,
            'coeffs': coeffs_cat,
            'residuals': res_cat,
            'x_filtered': x_cat,
            'y_filtered': y_cat
        })

    # Ellipse fitting
    h_guess = np.median(xs)
    k_guess = np.median(ys)
    a_guess = (np.max(xs) - np.min(xs)) / 2
    b_guess = (np.max(ys) - np.min(ys)) / 2
    model_func = left_ellipse if side == "left" else right_ellipse
    p0_ell = [h_guess, k_guess, a_guess, b_guess]
    
    h_lower = roi_x if side == "left" else roi_x + 0.65*roi_width
    h_upper = roi_x + 0.35*roi_width if side == "left" else roi_x + roi_width
    bounds_ell = (
        [h_lower, np.min(ys), 1e-3, 1e-3],
        [h_upper, np.max(ys), np.inf, np.inf]
    )
    
    x_ell, y_ell, coeffs_ell, res_ell = model_outlier_filter(
        xs, ys, model_func, p0_ell, bounds_ell
    )
    
    if coeffs_ell is not None:
        models.append({
            'type': 'ellipse',
            'func': model_func,
            'coeffs': coeffs_ell,
            'residuals': res_ell,
            'x_filtered': x_ell,
            'y_filtered': y_ell
        })

    # Select best model using RMSE
    for model in models:
        if model['residuals'] is not None:
            model['rmse'] = np.sqrt(np.mean(model['residuals']**2))
        else:
            model['rmse'] = np.inf
            
    return sorted(models, key=lambda x: x['rmse'])

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
    edges = cv2.Canny(blurred, 30, 100)
    
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
    for side_idx, (points, color, side) in enumerate(zip(
        [left_points, right_points], 
        [(0, 255, 255), (0, 165, 255)],
        ["left", "right"]
    )):
        if len(points) < 5:
            curves.append(None)
            continue
            
        xs, ys = points[:, 0], points[:, 1]
        models = fit_models(xs, ys, side, x_roi, w)
        
        if not models:
            curves.append(None)
            continue
            
        # Select best model
        best_model = models[0]
        curves.append({
            'type': best_model['type'],
            'func': best_model['func'],
            'coeffs': best_model['coeffs'],
            'color': color,
            'all_models': models  # Store all models for reporting
        })
        
        # Draw best fit curve
        y_vals = np.linspace(y_roi, y_roi+h, 100)
        x_vals = best_model['func'](y_vals, *best_model['coeffs'])
        pts = np.column_stack([x_vals, y_vals]).astype(int)
        cv2.polylines(output, [pts], False, color, 2)
            
    

    # Calculate coordinate system if both curves exist
    if all(curves) and len(curves) == 2:
        y_waist = find_waist(np.linspace(y_roi, y_roi+h, 1000), curves[0], curves[1])
        x_left = curves[0]['func'](y_waist, *curves[0]['coeffs'])
        x_right = curves[1]['func'](y_waist, *curves[1]['coeffs'])
        origin = ((x_left + x_right)/2, y_waist)
        
        # Create rotation matrix (vertical x-axis)
        theta = np.radians(-90)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
        
        # Draw coordinate system
        axis_length = 50
        cv2.line(output, (int(origin[0]), int(origin[1])),
                (int(origin[0] + axis_length*np.sin(theta)), 
                 int(origin[1] + axis_length*np.cos(theta))), (0, 0, 255), 2)

    # Process measurements
    output_data = []
    intersection_data = []
    
    for idx, curve in enumerate(curves):
        if not curve:
            continue
            
        try:
            # Get plate intersection points
            y_top = y_roi
            y_bottom = y_roi + h
            func = curve['func']
            x_top = func(y_top, *curve['coeffs'])
            x_bottom = func(y_bottom, *curve['coeffs'])
            
            curve["top_point"] = (x_top, y_top)
            curve["bottom_point"] = (x_bottom, y_bottom)
            
            if rotation_matrix is not None:
                curve["top_angle"] = calculate_contact_angle(curve, y_top, rotation_matrix)
                curve["bottom_angle"] = calculate_contact_angle(curve, y_bottom, rotation_matrix)
                
                top_tf = transform_coordinates(x_top, y_top, origin, rotation_matrix)
                bottom_tf = transform_coordinates(x_bottom, y_bottom, origin, rotation_matrix)
                
                intersection_data.extend([
                    {"type": "top", "side": idx, "original": (x_top, y_top), "transformed": top_tf},
                    {"type": "bottom", "side": idx, "original": (x_bottom, y_bottom), "transformed": bottom_tf}
                ])
            
            output_data.append({
                "side": "left" if idx == 0 else "right",
                "all_models": curve.get('all_models', []),  # Ensure this exists
                # ... rest of existing fields ...
                "coefficients": curve['coeffs'],
                "top_angle": curve.get('top_angle', None),
                "bottom_angle": curve.get('bottom_angle', None),
                "top_point": (x_top, y_top),
                "bottom_point": (x_bottom, y_bottom)
        })
            
        except Exception as e:
            print(f"Measurement error: {str(e)}")
            continue

    # Save results
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
            # Check if 'all_models' exists and is iterable
            models_to_report = data.get('all_models', [])
            if not isinstance(models_to_report, list):
                models_to_report = []
                
            for model in models_to_report:
                # Add null checks for coefficients
                coeffs = model.get('coeffs', [])
                if len(coeffs) < 4:  # Pad with N/A for ellipse params if needed
                    coeffs = list(coeffs) + ['N/A']*(4-len(coeffs))
                writer.writerow([
                    os.path.basename(image_path),
                    data["side"],
                    model['type'],
                    f"{model['rmse']:.4f}",
                    *model['coeffs'],
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