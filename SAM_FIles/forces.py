import numpy as np
import matplotlib.pyplot as plt


def calculate_force(H, y, theta, pixel_to_meter, sigma= 72):
    """Calculate force component using curvature with pixel conversion
    H: mean curvature in 1/m
    y: Capillary neck radius in pixels
    theta: angle in radians
    pixel_to_meter: conversion factor from pixels to meters
    sigma: surface tension in mN/m
    Returns: force in µN
    """
    y_m = y * pixel_to_meter
    H_m = np.abs(H / pixel_to_meter)
    theta = np.radians(theta)  # Convert angle to radians if needed
    angle_portion = -2 * np.pi * sigma * y_m * np.sin(theta) * 1e3
    curvature_portion = -2 * np.pi * sigma *(H_m / 2) * y_m**2 * 1e3
    # print(f"Angle portion: {angle_portion}, Curvature portion: {curvature_portion}")
    return angle_portion - curvature_portion

def get_H_from_nodoid_or_unduloid(a, bridge_type):
    if bridge_type == 'unduloid':
        return -1/np.abs(a)
    if bridge_type == 'nodoid' or 'catenoid':
        return 1/np.abs(a)
    

def calculate_mean_curvature(points, idx, kappa):
    x0 = points[0][idx]
    y0 = points[1][idx]
    # print(x0, y0, kappa)
    if kappa is not None:
        H = (kappa + 1/y0) / 2
        return H
    else: 
        return None
    
def numerical_kappa(points, idx):
    """
    Compute curvature at a point index on a parameterized 2D curve.
    points: Nx2 array of (x, y)
    idx: index at which to evaluate curvature
    Returns: scalar curvature κ
    """
    x = points[0]
    y = points[1]
    if idx < 2 or idx > len(x) - 3:
        # print("Index out of bounds for curvature calculation.")
        return None  # avoid edges

 # 1st derivatives (central difference)
    dx = (-x[idx + 2] + 8*x[idx + 1] - 8*x[idx - 1] + x[idx - 2]) / 12
    dy = (-y[idx + 2] + 8*y[idx + 1] - 8*y[idx - 1] + y[idx - 2]) / 12

    # 2nd derivatives (central difference)
    ddx = (-x[idx + 2] + 16*x[idx + 1] - 30*x[idx] + 16*x[idx - 1] - x[idx - 2]) / 12
    ddy = (-y[idx + 2] + 16*y[idx + 1] - 30*y[idx] + 16*y[idx - 1] - y[idx - 2]) / 12

    numerator = dx * ddy - dy * ddx
    denominator = (dx**2 + dy**2)**1.5

    if denominator == 0:
        print("Denominator zero — curvature undefined.")
        return None
    kappa = numerator / denominator
    return kappa

def new_curvature_calculation(points, idx):
    """
    Compute curvature at a point index on a parameterized 2D curve.
    points: Nx2 array of (x, y)
    idx: index at which to evaluate curvature
    Returns: scalar curvature κ
    """
    x = points[0]
    y = points[1]
    
    if idx < 2 or idx > len(x) - 3:
        return None  # avoid edges

    dydx = np.gradient(y, x)
    d2ydx2 = np.gradient(dydx, x)

    
    H = d2ydx2[idx] / (1 + dydx[idx]**2)**(3/2) - 1/(y[idx] * (1 + dydx[idx]**2)**(1/2))

    return H/2

def calculate_total_force_with_circle_model(contact_pt, origin, angle_deg, R2_px, pixel_to_meter, y_star, sigma=72):
    """
    contact_pt: (x, y) contact point in pixels
    origin: (x, y) bridge midpoint in pixels
    angle_deg: contact angle in degrees
    R2_px: radius of fitted mid-arc circle (in pixels)
    pixel_to_meter: conversion factor
    sigma: surface tension in mN/m
    Returns: total force in μN
    """
    if any(x is None for x in (contact_pt, origin, angle_deg, R2_px)):
        return None

    gamma = sigma  # mN/m
    R  = np.abs((np.array(contact_pt)[0] - np.array(origin)[0]))* pixel_to_meter  # convert to meters
    R1 = y_star * pixel_to_meter  # contact radius
    R2 = - R2_px * pixel_to_meter  # external curvature (concave bridge = negative)
    # print(f"R: {R}, R1: {R1}, R2: {R2}, angle_deg: {angle_deg}")
    theta = np.abs(np.radians(angle_deg))
    delta_P = gamma * (1 / R1 + 1 / R2)
    Fl = 2 * np.pi * R * gamma * np.sin(theta)
    Fp = -np.pi * R**2 * delta_P
    # print(f"Fl: {Fl}, Fp: {Fp}")
    F_total = Fl + Fp

    return F_total * 1e3  # μN
