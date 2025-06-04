import numpy as np
import matplotlib.pyplot as plt


def calculate_force(sigma, H, y, theta, pixel_to_meter):
    """Calculate force component using curvature with pixel conversion"""
    y_m = y * pixel_to_meter
    return -2 * np.pi * (
        sigma * (y_m * np.sin(theta) - (H/2) * y_m**2)
    ) * 1e3  # to µN

def get_H_from_nodoid_or_unduloid(a, bridge_type):
    if bridge_type == 'unduloid':
        return -1/np.abs(a)
    if bridge_type == 'nodoid' or 'catenoid':
        return 1/np.abs(a)
    

def numerical_kappa(points, idx):
    """
    Compute curvature at a point index on a parameterized 2D curve.
    points: Nx2 array of (x, y)
    idx: index at which to evaluate curvature
    Returns: scalar curvature κ
    """
    if idx < 2 or idx > len(points) - 3:
        return None  # avoid edges

    x0, y0 = points[idx]
    x1, y1 = points[idx - 1]
    x2, y2 = points[idx + 1]

    dx = x2 - x1
    dy = y2 - y1

    ddx = points[idx + 1, 0] - 2 * x0 + points[idx - 1, 0]
    ddy = points[idx + 1, 1] - 2 * y0 + points[idx - 1, 1]

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2) ** 1.5

    if denominator == 0:
        return None

    kappa = numerator / denominator
    return kappa

def calculate_mean_curvature(points, idx, kappa):
    x0 = points[0][idx]
    y0 = points[1][idx]
    kappa = np.full_like(x0, kappa)  # Ensure kappa is an array of the same length as y0
    H = (kappa + 1/y0) / 2
    plt.plot(x0, H, label='Mean Curvature H')
    plt.xlabel('x')
    plt.ylabel('Mean Curvature H')
    plt.title('Curvature along the Curve')
    plt.legend()
    plt.show()
    return np.mean(H)
