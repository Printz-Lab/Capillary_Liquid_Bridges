from numpy.linalg import svd, qr, solve, det
from numpy import array, matmul, float64, transpose # , allclose, matmul, float64
from math import sqrt
import cv2
import numpy as np
import os
import csv
import random
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import filedialog



SMALL = 10**-12
sqrt2 = 2**0.5
TYPE_HYPERBOLA = "hyperbola"
TYPE_PARABOLA = "parabola"
TYPE_ELLIPSE = "ellipse"
TYPE_LINEAR = "linear"
def centered(points) :
    """Subtract the average from the (x,y) pairs."""
    xs_uncentered = [x for (x,y) in points]
    ys_uncentered = [y for (x,y) in points]

    x_center = sum([x for x in xs_uncentered])/len(points)
    y_center = sum([y for y in ys_uncentered])/len(points)

    xs = [x-x_center for x in xs_uncentered]
    ys = [y-y_center for y in ys_uncentered]

    return list(zip(xs,ys)), (x_center, y_center)

def translate_conic(conic, displacement) :
    """Return a conic that has been translated by an amount (dx,dy)."""
    (dx, dy) = displacement
    A,B,C,D,E,F = conic.get("coeffs")
    new_coeffs = (A,B,C,
                  D - 2*A*dx - B*dy,
                  E - 2*C*dy - B*dx,
                  A*dx*dx + B*dx*dy + C*dy*dy - D*dx - E*dy + F)
    conic["coeffs"] = new_coeffs
    conic["center"] = np.array(conic.get("center",(0,0)))+ np.array(displacement)
    return conic

def collinearity(points):
    """
    Return a number indicating how nearly the points follow a line (0 = perfect).
    Also return the best-fit line in point-slope form: (dx, dy), (x0, y0).
    """

    (x0, y0) = points[0]

    # Check for vertical cluster first
    xs = [x for (x,y) in points]
    if max(xs) - min(xs) < SMALL:
        # Perfect vertical line
        return 0.0, (0.0, 1.0), (x0, y0)

    # Compute average x-deviation to detect near-vertical
    avg_dev = sum(abs(x - x0) for (x,y) in points[1:]) / (len(points)-1)
    if avg_dev < SMALL:
        # Really vertical
        return 0.0, (0.0, 1.0), (x0, y0)

    # Build list of finite slopes
    slopes = []
    for (x, y) in points[1:]:
        dx = x - x0
        if abs(dx) < SMALL:
            continue
        slopes.append((y - y0) / dx)

    if not slopes:
        # Fallback to vertical if no finite slope
        return 0.0, (0.0, 1.0), (x0, y0)

    slopes.sort()
    median_slope = slopes[len(slopes)//2]

    # Compute intercepts
    intercepts = [y - median_slope * x for (x,y) in points[1:]]
    intercepts.sort()
    median_intercept = intercepts[len(intercepts)//2]

    # Compute mean squared deviation
    mse = sum((y - median_slope*x - median_intercept)**2 for (x,y) in points) / len(points)

    # Return (fit_error, (dx,dy), (x0_line, y0_line))
    # Here the line direction vector is (1, median_slope), base point at x=0 => y=median_intercept
    return mse, (1.0, median_slope), (0.0, median_intercept)

def smallest_singular(M, compute_singular_vector=True) :
    """Return a pair containing the smallest singular value of M and a
corresponding singular vector.
    """

    if compute_singular_vector: 
        _, s, Vh = svd(M, full_matrices=True)
        return s[-1], Vh[-1]

def fit_conic(points, desired_type=None) :
    points, center = centered(points)

    matrix_monomials = array([
        [1, x, y,    y*y-x*x, 2*x*y, y*y+x*x]
        for (x,y) in points])

    # ---- Check for degenerate conic lying on a single line
    fit, slope, intercept = collinearity(points)
    # print("COLLINEARITY", fit, slope, intercept)
    if (fit < SMALL) :
        return translate_conic({"type" : TYPE_LINEAR,
                                "coeffs" : (0, 0, 0, slope[1], -slope[0], slope[0]*intercept[1] - slope[1]*intercept[0]),
                                "fit" : fit,        
                                }, center)

        return None


    # ---- Perform decomposition
    submatrix = matrix_monomials[:,:3] # (1,x,y)
    q,r = qr(submatrix, "complete")


    R_full = matmul(transpose(q), matrix_monomials)
    RA = R_full[:3, 0:3]
    RB = R_full[:3, 3:6]
    RC = R_full[3:, 3:6]
    _, vec_smallest = smallest_singular(RC)

    # ----- Determine type of conic
    Z = array([[-1,0,1],
               [0,sqrt2,0],
               [1,0,1]]) / sqrt2
    D = array([[-1,0,0],
               [0,-1,0],
               [0,0,1]])/4

    A_det = transpose(vec_smallest) @ D @ vec_smallest
    A_trace = vec_smallest[2]

    conic_type = TYPE_PARABOLA if (abs(A_det) < SMALL) else TYPE_ELLIPSE if A_det > 0 else TYPE_HYPERBOLA


    # ---- Adjust answer if it doesn't match desired conic type (not
    # ---- implemented yet)

    if(False and desired_type == TYPE_PARABOLA) :
        # "desired type" not implemented yet.  also not necessary for
        # this application b/c empirical points are expected to be
        # extremely unambiguously close to particular conics.

        pass
    else :
        q = vec_smallest


    # ------ Solve for conic coefficients
    ret = solve(RA, -1 * (RB @ q)) # c, *two_b

    params = [*ret, *q]

    conic_coeffs = [ params[5]-params[3],
                         2*params[4],
                         params[5]+params[3],
                         params[1],
                         params[2],
                         params[0]]


    fit  = sum([x**2 for x in matmul(matrix_monomials, array(params))])/len(points)


    A,B,C,D,E,F = conic_coeffs


    quadratic_form = array([[A, B/2],[B/2,C]])

    full_form = array([[  A, B/2, D/2],
                       [B/2,   C, E/2],
                       [D/2, E/2,   F]])

    if(abs(det(full_form)) < 1e-8) :
            # degenerate conic
            return None


    u,s,vh = svd(quadratic_form)

    K =  -det(full_form)/det(quadratic_form) if det(quadratic_form) != 0 else None

    axes = None
    try :
        axes = [(K/x)**0.5 for x in s.tolist()]
    except Exception :
        pass

    return translate_conic({"type" : conic_type,
                            "coeffs" : conic_coeffs,
                            "fit" : fit,
                            "axes" : axes,
                            "diagonalizer" : vh.tolist()       
                            }, center)

def newton_root(f, init_guess=60) :
    x = init_guess
    small = SMALL
    for _ in range(30) :
        if abs(f(x)) < 1e-8 :
            break
        else :
            x -= f(x)*small/(f(x+small)-f(x))
    return x

def select_roi(image):
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    return roi

def ransac_filter_std(points,
                      k=1.0,         # “σ‑multiplier”: inliers = dist < k * σ
                      n_trials=200,  
                      sample_size=6):
    """
    RANSAC based on per‑trial standard deviation of algebraic distance.

    points:       list of (x,y) tuples
    k:            keep points with distance < k * sigma
    n_trials:     how many random subsets to try
    sample_size:  minimal pts needed for a conic fit

    Returns: (x_inliers, y_inliers)
    """
    best_inliers = points
    best_count   = 0

    for _ in range(n_trials):
        # 1) pick a minimal sample and fit
        sample = random.sample(points, sample_size)
        model  = fit_conic(sample)
        if model is None:
            continue
        A,B,C,D,E,F = model['coeffs']

        # 2) compute all algebraic distances
        #    normalized so “1 unit” ~ 1px in image space
        norm = np.sqrt(A*A + (B/2)**2 + C*C)
        if norm < SMALL:
            continue

        dists = []
        for x,y in points:
            d = abs(A*x*x + B*x*y + C*y*y + D*x + E*y + F) / norm
            dists.append(d)
        sigma = np.std(dists)

        # 3) inliers = those within k*σ
        inliers = [pt for pt, d in zip(points, dists) if d < k*sigma]
        if len(inliers) > best_count:
            best_count   = len(inliers)
            best_inliers = inliers

    # if RANSAC found nothing better, fall back to original
    if best_count < sample_size:
        xs, ys = zip(*points)
    else:
        xs, ys = zip(*best_inliers)

    return np.array(xs), np.array(ys)

def get_intersections_with_y(conic, y_val):
    """Solve for x at a given y, with numerical stability checks."""
    A, B, C, D, E, F = conic['coeffs']
    a = A
    b = B * y_val + D
    c = C * (y_val**2) + E * y_val + F

    # Handle near-linear cases first
    if abs(a) < 1e-10:
        if abs(b) < 1e-10:
            return []  # No solution
        x = -c / b
        return [x]
    
    disc = b**2 - 4*a*c
    if disc < 0:
        return []  # No real roots
    
    sqrt_d = np.sqrt(disc)
    # Use numerically stable quadratic formula
    if b >= 0:
        x1 = (-b - sqrt_d) / (2*a)
        x2 = (2*c) / (-b - sqrt_d)
    else:
        x1 = (2*c) / (-b + sqrt_d)
        x2 = (-b + sqrt_d) / (2*a)
    
    # Filter valid solutions (within image bounds)
    valid = []
    for x in [x1, x2]:
        if 0 <= x < 10000:  # Adjust based on your image width
            valid.append(x)
    return valid

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    roi = select_roi(image)
    if roi == (0, 0, 0, 0):
        return

    x0_roi, y0_roi, w, h = roi
    cv2.rectangle(output, (x0_roi, y0_roi), (x0_roi + w, y0_roi + h), (0, 255, 0), 2)

    roi_img = image[y0_roi:y0_roi + h, x0_roi:x0_roi + w]
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 225, 250)

    edge_points = np.column_stack(np.where(edges > 0))
    if len(edge_points) < 10:
        print("Not enough edge points found")
        return

    abs_points = edge_points + [y0_roi, x0_roi]
    abs_points = abs_points[:, [1, 0]]  # (x, y)

    # Split into left/right regions
    left_mask = abs_points[:, 0] < x0_roi + w * .4
    right_mask = abs_points[:, 0] > x0_roi + w * 0.6
    sides = [('left', left_mask, (0, 255, 0)), ('right', right_mask, (0, 0, 255))]

    output_data = []

    for side, mask, color in sides:
        pts =abs_points[mask]
        if len(pts) < 50:
            print(f"Skipping {side}, only {len(pts)} pts")
            continue

        xs, ys = pts[:,0], pts[:,1]
        print(f"\n{side.capitalize()} side: {len(xs)} raw pts")

        # --- 1) RANSAC outlier removal ---
        pts_list = list(zip(xs, ys))
        x_filt, y_filt = ransac_filter_std(
            pts_list,
            k=2,    # adjust for your noise
            n_trials=500,     # more trials → more robu st
            sample_size=6     # conic needs ≥6 pts
        )
        print(f"RANSAC kept {len(x_filt)}/{len(xs)} pts")

        # --- 2) Fit conic on inliers ---
        inlier_pts = list(zip(x_filt, y_filt))
        conic = fit_conic(inlier_pts)
        if conic is None:
            print(f"Conic fit failed on {side}")
            continue

        # Draw raw vs inliers
        for x,y in zip(xs, ys):
            cv2.circle(output, (int(x),int(y)), 1, (200,200,0), -1)
        for x,y in zip(x_filt, y_filt):
            cv2.circle(output, (int(x),int(y)), 2, (0,200,200), -1)

        # Compute contact‐angle at top/bottom
        A,B,C,D,E,F = conic['coeffs']   
        pts_arr = np.array(inlier_pts)
        y_min, y_max = pts_arr[:,1].min(), pts_arr[:,1].max()

        # find mean x at those ys
        x_top = pts_arr[pts_arr[:,1]==y_min][:,0].mean()
        x_bot = pts_arr[pts_arr[:,1]==y_max][:,0].mean()

        def grad(x,y):
            return np.array([2*A*x + B*y + D,
                             B*x + 2*C*y + E])

        g_top = grad(x_top, y_min)
        g_bot = grad(x_bot, y_max)

        def adjust_angle(angle):
            return abs(180 - abs(angle)) if abs(angle) > 90 else angle

        angle_top = adjust_angle(np.degrees(np.arctan2(-g_top[0], g_top[1])))
        angle_bot = adjust_angle(np.degrees(np.arctan2(-g_bot[0], g_bot[1])))

        output_data.append({
            'side': side,
            'conic': conic,
            'inliers': inlier_pts,
            'raw_top': (x_top, y_min),
            'raw_bot': (x_bot, y_max),
            'top_angle': angle_top,
            'bottom_angle': angle_bot
        })


        # Mark contact points
        # cv2.circle(output, (int(x_top), int(y_min)), 4, (255,0,0), -1)
        # cv2.circle(output, (int(x_bot), int(y_max)), 4, (0,255,255), -1)

    if len(output_data) == 2:
        left, right = output_data

        # helper to solve each branch at a fixed y
        def branch_x(c, y, side):
            A,B,C,D,E,F = c['conic']['coeffs']
            a, b, c0 = A, B*y + D, C*y*y + E*y + F
            disc = b*b - 4*a*c0
            if disc < 0 or abs(a) < SMALL:
                return None
            sqrt_d = np.sqrt(disc)
            roots = [(-b + sqrt_d) / (2*a), (-b - sqrt_d) / (2*a)]
            return min(roots) if side == 'left' else max(roots)

        # 1) find the closest pair between the two inlier sets
        left_pts  = left['inliers']   # list of (x,y)
        right_pts = right['inliers']

        min_d2 = float('inf')
        best   = None
        for (xl, yl) in left_pts:
            for (xr, yr) in right_pts:
                d2 = (xr - xl)**2 + (yr - yl)**2
                if d2 < min_d2:
                    min_d2 = d2
                    best   = ((xl, yl), (xr, yr))

        if best is None:
            print("No valid inlier pairs found; skipping origin")
            return

        (xl, yl), (xr, yr) = best
        # 2) origin at midpoint of that pair:
        x0 = 0.5 * (xl + xr)
        y0 = 0.5 * (yl + yr)
        origin = (x0, y0)
        print(f"Origin at midpoint of closest inliers: ({x0:.2f}, {y0:.2f})")

        # 3) draw vertical (x-axis) and horizontal (y-axis)
        h_img, w_img = output.shape[:2]
        cv2.line(output, (int(x0), 0),        (int(x0), h_img), (255,255,255), 1)
        cv2.line(output, (0,        int(y0)), (w_img,  int(y0)), (255,255,255), 1)

        # 4) transform conic coefficients, intersections, closest point, angles
        for d in (left, right):
            side = d['side']
            A,B,C,D,E,F = d['conic']['coeffs']

            # 4a) transform coeffs into (u=y-y0, v=x-x0) frame
            #    original: A x^2 + B x y + C y^2 + D x + E y + F = 0
            #    new:      A2 v^2 + B2 v u + C2 u^2 + D2 v + E2 u + F2 = 0
            A2 = C
            B2 = B
            C2 = A
            D2 = 2*A*x0 + B*y0 + D
            E2 = 2*C*y0 + B*x0 + E
            F2 = A*x0*x0 + B*x0*y0 + C*y0*y0 + D*x0 + E*y0 + F
            d['trans_coefs'] = (A2, B2, C2, D2, E2, F2)

            # Replace the fallback code in your processing loop with:
            y_edge = y0_roi + h
            xs_bot = []
            if not xs_bot:
                # Find closest y with valid intersection
                search_ysteps = np.linspace(y_max, y_edge, 50)
                for y_search in search_ysteps:
                    xs_test = get_intersections_with_y(conic, y_search)
                    if xs_test:
                        x_bot_plate = max(xs_test) if side == 'right' else min(xs_test)
                        bottom_pt = (x_bot_plate, y_search)
                        break
                else:  # Fallback to parameteric conic equation
                    t = np.linspace(0, 2*np.pi, 100)
                    # Use parametric form based on conic type
                    if conic['type'] == TYPE_HYPERBOLA:
                        x_vals = conic['axes'][0] * np.cosh(t)
                        y_vals = conic['axes'][1] * np.sinh(t)
                    else:  # Ellipse
                        x_vals = conic['axes'][0] * np.cos(t)
                        y_vals = conic['axes'][1] * np.sin(t)
                    # Rotate and translate points
                    rotated = np.vstack([x_vals, y_vals]).T @ conic['diagonalizer']
                    translated = rotated + conic['center']
                    # Find point closest to plate
                    plate_y = y_edge
                    idx = np.argmin(np.abs(translated[:,1] - plate_y))
                    bottom_pt = (translated[idx,0], translated[idx,1])
                    
            if xs_bot:
                xs_bot_plate = max(xs_bot) if side == 'right' else min(xs_bot)
                bottom_pt = (x_bot_plate, y_edge)
                print(f"[OK] Conic intersects plate at {bottom_pt}")
            else:
                # Fallback 1: Tangent extrapolation
                try:
                    m = -g_bot[0] / g_bot[1]
                    x_bot_plate = x_bot + ((y_edge) - y_max) / m
                    bottom_pt = (x_bot_plate, y_edge)
                    print(f"[FALLBACK 1] Tangent extrapolation: {bottom_pt}")
                except:
                    # Fallback 2: Use bottom-most inlier point
                    bottom_pt = (xs_bot, y_max)
                    print(f"[FALLBACK 2] Using bottom-most inlier: {bottom_pt}")

            # 3) Now append everything in one go (preserving 'conic')
            output_data.append({
                'side':         side,
                'conic':        conic,
                'inliers':      inlier_pts,
                'raw_top':      (x_top,    y_min),
                'raw_bot':      bottom_pt,
                'top_angle':    angle_top,
                'bottom_angle': angle_bot
            })

            # 4c) find the closest point on the branch to the origin, raw & transformed
            best_dist2 = np.inf
            best_pt    = None
            for y in ys:
                x = branch_x(d, y, side)
                if x is None:
                    continue
                dx, dy = x - x0, y - y0
                d2 = dx*dx + dy*dy
                if d2 < best_dist2:
                    best_dist2 = d2
                    best_pt    = (x, y)
            if best_pt is not None:
                xc, yc = best_pt
                d['closest_raw']   = (xc, yc)
                d['closest_trans'] = (yc - y0, xc - x0)
            else:
                d['closest_raw']   = None
                d['closest_trans'] = None

        # 5) compute neck width
            neck_width = (xr - xl)/2

            # 6) draw & label intersection points and contact angles
            for side, d in (('left', left), ('right', right)):
                # colors: top=red, bottom=blue
                for label, col in (('top', (0,0,255)), ('bottom', (255,0,0))):
                    pt = d.get(f'raw_{label}')
                    angle = d.get(f'{label}_angle') or d.get(f'{label}_angle', d.get(f'{label}_angle'))
                    if pt:
                        x_i, y_i = pt
                        # draw intersection
                        cv2.circle(output, (int(x_i), int(y_i)), 6, col, -1)
                        # label intersection coords
                        txt = f"{side}_{label}: ({x_i:.1f},{y_i:.1f})"
                        cv2.putText(output, txt, (int(x_i)+8, int(y_i)-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                        # label contact angle
                        ang = d[f'{label}_angle']
                        txt2 = f"{ang:.1f}°"
                        cv2.putText(output, txt2, (int(x_i)+8, int(y_i)+12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # 7) draw & label neck width at the origin
        ox, oy = origin
        txt = f"Neck width: {neck_width:.2f}px"
        cv2.putText(output, txt, (int(ox)+10, int(oy)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)


                    # After conic fitting, add:
        if True:  # Debug visualization
            # Draw predicted conic
            t = np.linspace(-3, 3, 100)
            if conic['type'] == TYPE_HYPERBOLA:
                x = conic['axes'][0] * np.cosh(t)
                y = conic['axes'][1] * np.sinh(t)
            else:
                x = conic['axes'][0] * np.cos(t)
                y = conic['axes'][1] * np.sin(t)
            
            # Transform points
            pts = np.vstack([x, y]).T @ conic['diagonalizer'] + conic['center']
            pts = pts.astype(int)
            
            # Draw on image
            for pt in pts:
                if 0 <= pt[0] < output.shape[1] and 0 <= pt[1] < output.shape[0]:
                    cv2.circle(output, tuple(pt), 1, (0,255,255), -1)
            
            cv2.imshow("Conic Debug", output)
            cv2.waitKey(1000)

        # 8) print all the values to console
        print("=== Results ===")
        print(f"Origin: ({ox:.2f}, {oy:.2f})")
        print(f"Neck width: {neck_width:.2f} px")
        for side, d in (('Left', left), ('Right', right)):
            for label in ('top', 'bottom'):
                pt = d.get(f'raw_{label}')
                ang = d[f'{label}_angle']
                print(f"{side} {label.capitalize()} Intersection: {pt}, Angle: {ang:.2f}°")

    # display
    cv2.imshow("Final Results", output)
    cv2.waitKey(0)
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