"""
Boundary extraction and analysis methods
"""
import numpy as np
import cv2

def boundary_by_angle_max(points_xy: np.ndarray, center=None, bins: int = 720) -> np.ndarray:
    """
    Extract boundary points by taking maximum radius point in each angular bin
    Returns: Nx2 array of boundary points (x,y)
    """
    XY = np.asarray(points_xy, float)
    if center is None:
        from .circle_fitting import fit_circle_pratt
        cx, cy, _ = fit_circle_pratt(XY[:,0], XY[:,1])
    else:
        cx, cy = center

    dx = XY[:,0] - cx
    dy = XY[:,1] - cy
    ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    r = np.hypot(dx, dy)

    bin_idx = np.floor(ang / (2*np.pi / bins)).astype(int)
    bin_idx = np.clip(bin_idx, 0, bins-1)

    border = []
    for b in range(bins):
        mask = (bin_idx == b)
        if not np.any(mask):
            continue
        i_local = np.argmax(r[mask])
        idxs = np.nonzero(mask)[0]
        i = idxs[i_local]
        border.append((XY[i,0], XY[i,1], ang[i]))
    
    if not border:
        raise RuntimeError("No boundary points extracted (data too sparse?)")
    
    border = np.array(border, float)
    order = np.argsort(border[:,2])
    return border[order][:,:2]

def boundary_by_convex_hull(points_xy: np.ndarray) -> np.ndarray:
    """
    Extract boundary points using convex hull
    Returns: Nx2 array of boundary points (x,y)
    """
    pts32 = points_xy.astype(np.float32).reshape(-1,1,2)
    hull = cv2.convexHull(pts32, returnPoints=True)
    edge_xy = hull.reshape(-1,2).astype(np.float64)
    if len(edge_xy) < 10:
        raise RuntimeError("Too few hull vertices")
    return edge_xy

def ovality_and_fourier(x: np.ndarray, y: np.ndarray, 
                       center_x: float, center_y: float, 
                       radius: float) -> dict:
    """
    Calculate ovality metrics and Fourier analysis of boundary
    Returns: dict with Rmax, Rmin, ovality metrics, k2 amplitude
    """
    ang = (np.arctan2(y-center_y, x-center_x) + 2*np.pi) % (2*np.pi)
    rad = np.hypot(x-center_x, y-center_y)
    dev = rad - radius
    
    order = np.argsort(ang)
    ang, dev, rad = ang[order], dev[order], rad[order]

    Rmax, Rmin = rad.max(), rad.min()
    oval_abs = Rmax - Rmin
    oval_pct = 100.0 * oval_abs / radius if radius > 0 else np.nan

    # Fourier analysis
    K = max(1024, len(dev))
    theta_u = np.linspace(0, 2*np.pi, K, endpoint=False)
    dev_u = np.interp(theta_u, ang, dev, period=2*np.pi)

    F = np.fft.rfft(dev_u) / (K/2)
    amps = np.abs(F)
    k2_amp = amps[2] if len(amps) > 2 else 0.0

    return {
        "Rmax": Rmax, "Rmin": Rmin,
        "ovality_abs": oval_abs,
        "ovality_pct": oval_pct,
        "k2_amp": k2_amp,
        "theta": theta_u, 
        "dev_uniform": dev_u
    }
