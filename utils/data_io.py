"""
Data loading and saving utilities
"""
import os
import numpy as np
import pandas as pd

def load_txt_points(path: str) -> np.ndarray:
    """Load points from text file with x,y,z coordinates"""
    pts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith(("#", "//", ";")):
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
                pts.append((x, y, z))
            except Exception:
                continue
    if not pts:
        raise RuntimeError("No valid XYZ lines found in TXT.")
    return np.asarray(pts, dtype=np.float64)

def save_results(results: list, points: list, output_dir: str):
    """Save analysis results and boundary points to CSV files"""
    if not results:
        return
    
    # Save per-slice results
    df_res = pd.DataFrame(results).sort_values("z_center").reset_index(drop=True)
    results_csv = os.path.join(output_dir, "slice_results.csv")
    df_res.to_csv(results_csv, index=False, float_format="%.6f")
    print(f"\nSaved results -> {results_csv}")
    print(df_res.head(10))

    # Save boundary points if available
    if points:
        df_pts = pd.concat(points, ignore_index=True)
        df_pts = df_pts.sort_values(["z_center","theta"]).reset_index(drop=True)
        points_csv = os.path.join(output_dir, "slice_points.csv")
        df_pts.to_csv(points_csv, index=False, float_format="%.6f")
        print(f"Saved per-edge points -> {points_csv}")
