"""
Visualization utilities for cylinder analysis
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_slice(slice_points: np.ndarray, 
              edge_points: np.ndarray,
              center: tuple,
              radius: float,
              z_info: dict,
              window_len: float,
              z_step: float,
              boundary_method: str,
              output_path: str,
              max_plot_points: int = 30000):
    """Plot single slice with points, boundary and fitted circle"""
    # Subsample points if needed
    if len(slice_points) > max_plot_points:
        idx = np.random.choice(len(slice_points), max_plot_points, replace=False)
        plot_points = slice_points[idx]
    else:
        plot_points = slice_points

    # Create plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    
    # Plot points
    ax.scatter(plot_points[:,0], plot_points[:,1], 
              s=0.2, alpha=0.15, label="slice points")
    
    # Plot boundary
    ax.plot(edge_points[:,0], edge_points[:,1], 'k-',
            linewidth=1, alpha=0.9,
            label=("convex hull" if boundary_method=='convex_hull' else "angle-max border"))
    
    # Plot fitted circle
    theta = np.linspace(0, 2*np.pi, 720)
    xc, yc = center
    circ_x = xc + radius*np.cos(theta)
    circ_y = yc + radius*np.sin(theta)
    ax.plot(circ_x, circ_y, linewidth=2, label="LM-fitted circle")
    
    # Plot center
    ax.scatter([xc], [yc], s=25, label="center")
    
    # Formatting
    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.legend()
    title = f"window=[{z_info['z_low']:.3f},{z_info['z_high']:.3f}] "
    title += f"len={window_len}  step={z_step}"
    ax.set_title(title)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_summary(results_df: dict, window_len: float, z_step: float):
    """Plot ovality and radius summary plots"""
    # Ovality plot
    plt.figure(figsize=(8,4.5))
    plt.plot(results_df["z_center"], results_df["ovality_pct"], 
            marker='o', linewidth=1)
    plt.xlabel("z center")
    plt.ylabel("Ovality (%)")
    plt.title(f"Ovality percent vs z (window={window_len}, step={z_step})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Radius plot
    plt.figure(figsize=(8,4.5))
    plt.plot(results_df["z_center"], results_df["R"], 
            marker='.', linewidth=1)
    plt.xlabel("z center")
    plt.ylabel("Radius (unit)")
    plt.title("Fitted radius vs z")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_overlay(results_df: dict, points_df: dict, output_path: str):
    """Create overlay plot of all fitted circles"""
    fig, ax = plt.subplots(1,1, figsize=(7,7))
    
    # Color mapping function
    zvals = results_df["z_center"].to_numpy()
    zmin_v, zmax_v = zvals.min(), zvals.max()
    def z2c(z):
        t = 0.0 if zmax_v==zmin_v else (z - zmin_v) / (zmax_v - zmin_v)
        return plt.cm.viridis(t)

    # Plot edge points if available
    if points_df is not None:
        max_pts_overlay = 30000
        if len(points_df) > max_pts_overlay:
            pts_plot = points_df.sample(max_pts_overlay, random_state=0)
        else:
            pts_plot = points_df
        colors_pts = [z2c(z) for z in pts_plot["z_center"].to_numpy()]
        ax.scatter(pts_plot["x"], pts_plot["y"], 
                  s=1, alpha=0.25, c=colors_pts, label="edges (sample)")

    # Plot circles
    theta = np.linspace(0, 2*np.pi, 360)
    for _, row in results_df.iterrows():
        cx, cy, R, zc = row["cx"], row["cy"], row["R"], row["z_center"]
        c = z2c(zc)
        ax.plot(cx + R*np.cos(theta), cy + R*np.sin(theta), 
               alpha=0.9, linewidth=1.2, c=c)

    ax.set_aspect('equal', 'box')
    ax.grid(True)
    ax.set_title(f"Overlay fitted circles")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
