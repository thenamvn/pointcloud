# ============================================================
# Hybrid circle fit at a Z-slice:
# ConvexHull (reduce) -> RANSAC (robust) -> Pratt (init) -> LM+Huber (refine)
# ============================================================

from __future__ import annotations
from typing import Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

# ---------- Utils: quick 2D rank check ----------
def _has_enough_2d_variation(xy: np.ndarray, eps: float = 1e-12) -> bool:
    if xy.shape[0] < 3:
        return False
    xc = xy - xy.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(xc, full_matrices=False)
    return s[1] > eps * (s[0] + eps)

# ---------- Circle from 3 points (for RANSAC) ----------
def _circle_from_3pts(p1, p2, p3):
    (x1,y1), (x2,y2), (x3,y3) = p1, p2, p3
    a = 2*(x2-x1); b = 2*(y2-y1); c = x2*x2+y2*y2 - x1*x1-y1*y1
    d = 2*(x3-x2); e = 2*(y3-y2); f = x3*x3+y3*y3 - x2*x2-y2*y2
    M = np.array([[a,b],[d,e]], float)
    rhs = np.array([c,f], float)
    if abs(np.linalg.det(M)) < 1e-12:
        return None
    cx, cy = np.linalg.solve(M, rhs)
    R = float(np.hypot(x1-cx, y1-cy))
    return float(cx), float(cy), float(R)

# ---------- Pratt algebraic initial fit ----------
def fit_circle_pratt(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 points for Pratt fit")
    xm, ym = x.mean(), y.mean()
    u, v = x - xm, y - ym
    Suu, Svv, Suv = np.sum(u*u), np.sum(v*v), np.sum(u*v)
    Suuu, Svvv = np.sum(u*u*u), np.sum(v*v*v)
    Suvv, Svuu = np.sum(u*v*v), np.sum(v*u*u)
    A = np.array([[2*Suu, 2*Suv],[2*Suv, 2*Svv]], float)
    b = np.array([Suuu + Suvv, Svvv + Svuu], float)
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = xm + uc, ym + vc
    R = np.sqrt(uc*uc + vc*vc + (Suu + Svv)/n)
    return float(cx), float(cy), float(R)

# ---------- LM + Huber geometric refine ----------
def lm_circle_geometric(
    x: np.ndarray, y: np.ndarray,
    cx0: float, cy0: float, r0: float,
    huber_delta: float = 1.0,
    max_iter: int = 100,
    lm_lambda0: float = 1e-2,
    tol_update: float = 1e-10,
    tol_cost: float = 1e-12,
) -> Tuple[float, float, float]:
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    cx, cy, R = float(cx0), float(cy0), float(r0)

    def residuals(cx, cy, R):
        dx = x - cx; dy = y - cy
        ri = np.sqrt(dx*dx + dy*dy) + 1e-16
        return ri - R, dx, dy, ri

    def huber_weights(r, delta):
        a = np.abs(r); w = np.ones_like(r)
        m = a > delta; w[m] = delta/(a[m] + 1e-16)
        return w

    lam = lm_lambda0
    cost_prev = np.inf
    for _ in range(max_iter):
        r, dx, dy, ri = residuals(cx, cy, R)
        w = huber_weights(r, huber_delta)
        J = np.stack([-dx/ri, -dy/ri, -np.ones_like(ri)], axis=1)
        sw = np.sqrt(w); JW = J*sw[:,None]; rW = r*sw
        H = JW.T @ JW
        g = JW.T @ rW
        H_lm = H + lam*np.diag(np.diag(H))
        try:
            dpar = -np.linalg.solve(H_lm, g)
        except np.linalg.LinAlgError:
            lam *= 3; continue
        cx_n, cy_n, R_n = cx + dpar[0], cy + dpar[1], max(1e-12, R + dpar[2])

        r_new, *_ = residuals(cx_n, cy_n, R_n)
        a = np.abs(r_new)
        m = a <= huber_delta
        cost = 0.5*np.sum(a[m]**2) + np.sum(huber_delta*(a[~m] - 0.5*huber_delta))

        if cost < cost_prev - tol_cost:
            cx, cy, R = cx_n, cy_n, R_n
            cost_prev = cost
            lam = max(lam/3, 1e-12)
            if np.linalg.norm(dpar) < tol_update:
                break
        else:
            lam *= 3
    return float(cx), float(cy), float(R)

# ---------- RANSAC on candidate points ----------
def _ransac_circle(
    xy: np.ndarray,
    inlier_tol: float = 0.5,
    max_trials: int = 4000,
    min_inliers_frac: float = 0.1,
    r_range: Optional[Tuple[float,float]] = None,
    rng_seed: int = 42
) -> Optional[Dict[str, np.ndarray]]:
    n = xy.shape[0]
    if n < 3:
        return None
    best = {"score": -1, "cx": None, "cy": None, "R": None, "mask": None}
    rng = np.random.default_rng(rng_seed)
    for _ in range(max_trials):
        idx = rng.choice(n, size=3, replace=False)
        c = _circle_from_3pts(xy[idx[0]], xy[idx[1]], xy[idx[2]])
        if c is None:
            continue
        cx, cy, R = c
        if r_range is not None:
            if not (r_range[0] <= R <= r_range[1]):
                continue
        dx = xy[:,0] - cx; dy = xy[:,1] - cy
        err = np.abs(np.sqrt(dx*dx + dy*dy) - R)
        mask = err <= inlier_tol
        score = int(mask.sum())
        if score > best["score"]:
            best.update(score=score, cx=cx, cy=cy, R=R, mask=mask)

    if best["score"] < max(int(min_inliers_frac*n), 3):
        return None
    return best

# ---------- HYBRID MAIN ----------
def fit_circle_hybrid_at_z(
    points_xyz: np.ndarray,
    z_elevation: float,
    z_tolerance: float,
    inlier_tol: float = 0.5,
    max_trials: int = 4000,
    min_inliers_frac: float = 0.1,
    r_range: Optional[Tuple[float,float]] = None,
    huber_delta: float = 1.0,
    use_hull_vertices_only: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Hybrid circle fit on a Z-slice:
      ConvexHull -> RANSAC -> Pratt -> LM+Huber

    Returns dict:
      {
        "center_x","center_y","radius",
        "inlier_mask",  # mask over the candidate set used in RANSAC
        "fit_points_xy",# the full slice points (for plotting/context)
        "candidates_xy" # points used for RANSAC (hull vertices or slice)
      } or None if slice degenerate.
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")

    # Z-slice
    z = points_xyz[:,2]
    m = (z >= z_elevation - z_tolerance) & (z <= z_elevation + z_tolerance)
    if not np.any(m):
        return None
    xy = points_xyz[m,:2]
    if xy.shape[0] < 3:
        return None

    # Deduplicate helps hull/RANSAC
    if xy.shape[0] >= 2:
        xy = np.unique(xy, axis=0)

    # Convex hull to reduce candidates
    candidates = None
    if xy.shape[0] >= 3 and _has_enough_2d_variation(xy):
        try:
            hull = ConvexHull(xy)
            hull_xy = xy[hull.vertices]  # CCW order
            if use_hull_vertices_only and hull_xy.shape[0] >= 3:
                candidates = hull_xy
        except QhullError:
            candidates = None

    if candidates is None:
        # fallback: use all slice points
        candidates = xy

    if candidates.shape[0] < 3 or not _has_enough_2d_variation(candidates):
        return None

    # RANSAC on candidates
    model = _ransac_circle(
        candidates,
        inlier_tol=inlier_tol,
        max_trials=max_trials,
        min_inliers_frac=min_inliers_frac,
        r_range=r_range
    )
    if model is None:
        return None

    # Pratt init on inliers (over candidates)
    inl_cand = candidates[model["mask"]]
    if inl_cand.shape[0] >= 3:
        cx0, cy0, R0 = fit_circle_pratt(inl_cand[:,0], inl_cand[:,1])
    else:
        cx0, cy0, R0 = model["cx"], model["cy"], model["R"]

    # LM+Huber refine trên inliers (ổn định hơn dùng tất cả)
    cx, cy, R = lm_circle_geometric(inl_cand[:,0], inl_cand[:,1], cx0, cy0, R0, huber_delta=huber_delta)

    return {
        "center_x": cx,
        "center_y": cy,
        "radius": R,
        "inlier_mask": model["mask"],      # over candidates
        "fit_points_xy": xy,               # all slice points
        "candidates_xy": candidates,       # candidate set used in RANSAC
        "z_elevation": float(z_elevation)
    }

# ---------- Visualization ----------
def visualize_hybrid_fit(res: Optional[Dict[str,np.ndarray]]):
    plt.figure(figsize=(7,7))
    if res is None:
        plt.title("No fit (degenerate slice)")
        plt.xlabel("X"); plt.ylabel("Y"); plt.gca().set_aspect("equal"); plt.show(); return
    xy = res["fit_points_xy"]
    cand = res["candidates_xy"]
    mask = res["inlier_mask"]
    cx, cy, R = res["center_x"], res["center_y"], res["radius"]

    # All slice points
    plt.scatter(xy[:,0], xy[:,1], s=4, alpha=0.25, label="Slice XY (all)")

    # Candidates (e.g., hull vertices)
    plt.scatter(cand[:,0], cand[:,1], s=20, alpha=0.8, label=f"Candidates (n={len(cand)})")

    # Inliers among candidates
    plt.scatter(cand[mask,0], cand[mask,1], s=24, edgecolor="k", linewidths=0.5, label=f"Inliers (n={mask.sum()})")

    # Fitted circle
    th = np.linspace(0, 2*np.pi, 720, endpoint=False)
    plt.plot(cx + R*np.cos(th), cy + R*np.sin(th), lw=2, label=f"Fit R={R:.4f}")
    plt.scatter([cx],[cy], c="red", s=40, label="Center")

    plt.gca().set_aspect("equal")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Hybrid: Hull → RANSAC → Pratt → LM(Huber)")
    plt.legend()
    plt.tight_layout()
    plt.show()
