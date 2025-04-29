import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_csv_data(input_dir):
    """Load and combine all CSV files in a directory"""
    csv_files = Path(input_dir).glob('*.csv')
    dfs = []
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['Source CSV'] = csv_file.name
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

def get_pixel_conversion_factor(filename):
    """Extract mm value from filename and calculate pixel-to-meter conversion"""
    try:
        base_name = filename.split('.')[0]
        mm_part = [s for s in base_name.split('_') if s.endswith('mm')][-1]
        mm_value = float(mm_part.replace('mm', '')) / 100000  # Convert to meters
        return mm_value
    except:
        return None

def get_catenary_derivatives(a, y0, c, side):
    """Create catenary functions and their derivatives"""
    if side == 'left':
        return {
            'X': lambda y: -a * np.cosh((y - y0)/a) + c,
            'dXdy': lambda y: np.sinh((y - y0)/a),
            'd2Xdy2': lambda y: (1/a) * np.cosh((y - y0)/a)
        }
    else:  # right
        return {
            'X': lambda y:  c + a * np.cosh((y - y0)/a),
            'dXdy': lambda y: np.sinh((y - y0)/a),
            'd2Xdy2': lambda y: (1/a) * np.cosh((y - y0)/a)
        }

def calculate_mean_curvature(a, y0, c, side, y_values):
    """Compute mean curvature for catenary profile"""
    funcs = get_catenary_derivatives(a, y0, c, side)
    X     = funcs['X'](y_values)
    dXdy  = funcs['dXdy'](y_values)
    d2Xdy2= funcs['d2Xdy2'](y_values)
    
    denom          = (1 + dXdy**2)**1.5
    curvature_term = d2Xdy2 / denom
    azimuthal_term = 1 / (X * np.sqrt(1 + dXdy**2))
    
    valid = (np.abs(X) > 1e-6) & (np.abs(denom) > 1e-6)
    H = np.full_like(X, np.nan)
    H[valid] = curvature_term[valid] + azimuthal_term[valid]
    return H

def calculate_force(sigma, H, y, theta, pixel_to_meter):
    """Calculate force component using curvature with pixel conversion"""
    y_m = y * pixel_to_meter
    return -2 * np.pi * (
        sigma * (y_m * np.sin(theta) - (H/2) * y_m**2)
    ) * 1e3  # to µN

def main():
    input_dir = r"C:\Users\ezrap\OneDrive\Documents\Spring 2025 HW\Printz Lab Research\Capillary Bridging\Anton's snapshots\IPTMS Filtered\analyzed_results"
    sigma     = 72  # mN/m
    y_range   = np.linspace(-20, 20, 100)
    
    df = load_csv_data(input_dir)
    results = []
    
    for side in ['left', 'right']:
        sub = df[df['Side'] == side]
        if sub.empty:
            continue
        
        plt.figure(figsize=(8,6))
        for _, row in sub.iterrows():
            mm_value = get_pixel_conversion_factor(row['Filename'])
            if mm_value is None:
                continue
            x_span = abs(row['Top Transformed X'] - row['Bottom Transformed X'])
            if x_span <= 0:
                continue
            px2m = mm_value / x_span
            
            # curvature
            H = calculate_mean_curvature(
                row['a'], row['y0'], row['c'],
                side, y_range
            )
            medH = np.nanmedian(H)
            
            # plot
            x_m = y_range * px2m
            plt.plot(x_m, H, label=row['Filename'])
            
            # forces
            F_top = calculate_force(
                sigma, medH,
                row['Top Transformed Y'],
                np.deg2rad(row['Top Angle']),
                px2m
            )
            F_bot = calculate_force(
                sigma, medH,
                row['Bottom Transformed Y'],
                np.deg2rad(row['Bottom Angle']),
                px2m
            )
            
            results.append({
                'Filename': row['Filename'],
                'Side': side,
                'Top Angle (degrees)': row['Top Angle'],
                'Bottom Angle (degrees)': row['Bottom Angle'],
                'Mean Curvature (1/m)': medH,
                'Top Force (µN)': F_top,
                'Bottom Force (µN)': F_bot,
                
            })
        
        # decorate & show per‐side plot
        plt.xlabel("x (m)")
        plt.ylabel("Mean curvature H (1/m)")
        plt.title(f"Mean Curvature vs. x for all {side.capitalize()} profiles")
        plt.grid(True)
        plt.legend(ncol=2, fontsize='small')
        plt.tight_layout()
        plt.show()
    
    # summary table
    results_df = pd.DataFrame(results)
    print(results_df)    
    output_path = Path(input_dir) / 'force_calculations_results.csv'
    results_df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    main()
