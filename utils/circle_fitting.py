"""
Circle fitting algorithms including Pratt and LM+Huber methods
"""
import numpy as np
from typing import Tuple

def fit_circle_pratt(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit circle using Pratt's method
    Returns: center_x, center_y, radius
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    Suu, Suv, Svv = np.sum(u*u), np.sum(u*v), np.sum(v*v)
    Suuu, Svvv = np.sum(u*u*u), np.sum(v*v*v)
    Suvv, Svuu = np.sum(u*v*v), np.sum(v*u*u)
    A = np.array([[Suu, Suv],[Suv, Svv]], float)
    b = 0.5*np.array([Suuu + Suvv, Svvv + Svuu], float)
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = np.linalg.pinv(A) @ b
    xc, yc = xm + uc, ym + vc
    R = np.sqrt(uc*uc + vc*vc + (Suu+Svv)/len(x))
    return xc, yc, R

def lm_circle_geometric(x: np.ndarray, y: np.ndarray, 
                       a0: float, b0: float, R0: float, 
                       max_iter: int = 50, 
                       huber_delta: float = None, 
                       lam0: float = 1e-3, 
                       tol: float = 1e-10) -> Tuple[float, float, float, float]:
    """
    Levenberg-Marquardt optimization for circle fitting with Huber loss
    Returns: center_x, center_y, radius, final_cost
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    a, b, R = float(a0), float(b0), float(R0)
    lam = lam0

    r = np.sqrt((x-a)**2 + (y-b)**2) + 1e-15
    e = r - R
    if huber_delta is None:
        mad = np.median(np.abs(e - np.median(e))) + 1e-15
        huber_delta = 1.4826 * mad if mad > 0 else np.percentile(np.abs(e), 70) + 1e-12

    def huber_weights(res, delta):
        absr = np.abs(res)
        w = np.ones_like(res)
        mask = absr > delta
        w[mask] = (delta / absr[mask])
        return w

    prev_cost = np.inf
    for _ in range(max_iter):
        r = np.sqrt((x-a)**2 + (y-b)**2) + 1e-15
        e = r - R
        w = huber_weights(e, huber_delta) if huber_delta > 0 else np.ones_like(e)

        Ja = -(x - a) / r
        Jb = -(y - b) / r
        JR = -np.ones_like(r)

        sw = np.sqrt(w)
        J = np.vstack((Ja*sw, Jb*sw, JR*sw)).T
        ew = e * sw

        H = J.T @ J
        g = J.T @ ew

        H_lm = H + lam * np.eye(3)
        try:
            step = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            step = -np.linalg.pinv(H_lm) @ g

        a_new, b_new, R_new = a + step[0], b + step[1], R + step[2]

        r_new = np.sqrt((x-a_new)**2 + (y-b_new)**2) + 1e-15
        e_new = r_new - R_new
        w_new = huber_weights(e_new, huber_delta) if huber_delta > 0 else np.ones_like(e_new)
        cost_new = np.sum(w_new * e_new**2)

        if cost_new < prev_cost - 1e-12:
            a, b, R = a_new, b_new, R_new
            prev_cost = cost_new
            lam = max(lam/3, 1e-12)
            if np.linalg.norm(step) < tol * (1 + np.linalg.norm([a,b,R])):
                break
        else:
            lam = min(lam*5, 1e12)

    return a, b, abs(R), prev_cost
