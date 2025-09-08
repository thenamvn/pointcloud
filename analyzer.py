"""
Main analysis module for processing cylinder point clouds
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass

from utils.data_io import load_txt_points, save_results  # Modified imports
from utils.circle_fitting import fit_circle_pratt, lm_circle_geometric
from utils.boundary import (boundary_by_angle_max, boundary_by_convex_hull, 
                          ovality_and_fourier)
from utils.visualization import plot_slice, plot_summary, plot_overlay

@dataclass
class Config:
    """Analysis configuration parameters"""
    window_len: float = 9.0
    z_step: float = 2.0
    max_points_for_speed: int = 1_000_000
    min_points_per_slice: int = 1000
    inlier_quantile: float = 0.80
    boundary_method: str = 'angle_max'
    angle_bins: int = 720
    draw_per_slice_images: bool = True
    slice_plot_sample_points: int = 30000
    overlay_all: bool = True

class CylinderAnalyzer:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.dz = self.config.window_len / 2.0
        
        # Output paths
        self.slice_fig_dir = 'plots_slices'
        self.results_csv = 'slice_results.csv'
        self.points_csv = 'slice_points.csv'
        self.overlay_png = 'all_circles_overlay.png'
        
        # Create output directory
        os.makedirs(self.slice_fig_dir, exist_ok=True)
        
    def process_slice(self, points: np.ndarray, z_center: float) -> Tuple[dict, pd.DataFrame, str]:
        """Process a single slice of points"""
        # Filter points in z-window
        mask = (points[:,2] >= z_center - self.dz) & (points[:,2] <= z_center + self.dz)
        slc = points[mask][:,:2]  # Only x,y
        n0 = len(slc)
        
        if n0 < self.config.min_points_per_slice:
            return None, None, f"SKIP zc={z_center:.3f}: too sparse ({n0} points)"
            
        if n0 > self.config.max_points_for_speed:
            idx = np.random.choice(n0, self.config.max_points_for_speed, replace=False)
            slc = slc[idx]
            n0 = len(slc)

        # Extract boundary
        try:
            if self.config.boundary_method == 'convex_hull':
                edge_xy = boundary_by_convex_hull(slc)
            else:
                edge_xy = boundary_by_angle_max(slc, center=None, 
                                              bins=self.config.angle_bins)
        except RuntimeError as e:
            return None, None, f"SKIP zc={z_center:.3f}: {e}"

        # Initial Pratt fit
        xc1, yc1, R1 = fit_circle_pratt(edge_xy[:,0], edge_xy[:,1])
        
        # Get inliers
        rho1 = np.hypot(edge_xy[:,0]-xc1, edge_xy[:,1]-yc1)
        res1 = np.abs(rho1 - R1)
        thr = np.quantile(res1, self.config.inlier_quantile)
        good = res1 <= thr
        
        # Refined fits
        xc2, yc2, R2 = fit_circle_pratt(edge_xy[good,0], edge_xy[good,1])
        xc3, yc3, R3, cost = lm_circle_geometric(edge_xy[good,0], edge_xy[good,1], 
                                                xc2, yc2, R2)

        # Calculate metrics
        dx = edge_xy[:,0] - xc3
        dy = edge_xy[:,1] - yc3
        rad = np.hypot(dx, dy)
        ang = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
        
        # Get extreme points
        i_max = int(np.argmax(rad))
        i_min = int(np.argmin(rad))
        x_rmax, y_rmax = float(edge_xy[i_max,0]), float(edge_xy[i_max,1])
        x_rmin, y_rmin = float(edge_xy[i_min,0]), float(edge_xy[i_min,1])

        # Ovality analysis
        ov = ovality_and_fourier(edge_xy[:,0], edge_xy[:,1], xc3, yc3, R3)
        
        # Compile results
        result = {
            "z_center": z_center,
            "z_low": z_center - self.dz, 
            "z_high": z_center + self.dz,
            "n_points_slice": int(n0),
            "n_edge": int(len(edge_xy)),
            "cx": xc3, "cy": yc3, "R": R3,
            "Rmax": ov["Rmax"], 
            "Rmin": ov["Rmin"],
            "x_at_Rmax": x_rmax, 
            "y_at_Rmax": y_rmax,
            "x_at_Rmin": x_rmin, 
            "y_at_Rmin": y_rmin,
            "ovality_abs": ov["ovality_abs"],
            "ovality_pct": ov["ovality_pct"],
            "k2_amp": ov["k2_amp"],
            "cost": cost
        }
        
        # Record boundary points
        edge_records = pd.DataFrame({
            "z_center": z_center,
            "x": edge_xy[:,0],
            "y": edge_xy[:,1],
            "radius": rad,
            "theta": ang,
            "inlier80": good.astype(int)
        })
        
        return (result, edge_xy, slc, (xc3, yc3, R3), ov), edge_records, None

    def analyze_file(self, file_path: str):
        """Process entire point cloud file"""
        # Load points
        points = load_txt_points(file_path)
        z_min, z_max = points[:,2].min(), points[:,2].max()
        print(f"Z range in data: [{z_min:.3f}, {z_max:.3f}]  "
              f"| window={self.config.window_len}  step={self.config.z_step}")
        
        # Define z-centers
        if z_max - z_min < self.config.window_len:
            z_centers = np.array([(z_min + z_max)/2.0])
        else:
            z_centers = np.arange(z_min + self.dz, 
                                z_max - self.dz + 1e-9, 
                                self.config.z_step)
        
        results = []
        point_rows = []
        
        # Process each slice
        for zc in z_centers:
            out, edge_df, warn = self.process_slice(points, zc)
            if warn is not None:
                print(warn)
                continue
                
            res, edge_xy, slc, (xc, yc, R), ov = out
            results.append(res)
            if edge_df is not None:
                point_rows.append(edge_df)
                
            # Plot slice if requested
            if self.config.draw_per_slice_images:
                out_png = os.path.join(self.slice_fig_dir, f"slice_zc={zc:.3f}.png")
                plot_slice(slc, edge_xy, (xc,yc), R, res, 
                          self.config.window_len, self.config.z_step,
                          self.config.boundary_method, out_png,
                          self.config.slice_plot_sample_points)
        
        # Save results
        if results:
            save_results(results, point_rows, ".")
            df_res = pd.DataFrame(results)
            
            # Generate summary plots
            plot_summary(df_res, self.config.window_len, self.config.z_step)
            
            # Generate overlay plot
            if self.config.overlay_all:
                df_pts = pd.concat(point_rows, ignore_index=True) if point_rows else None
                plot_overlay(df_res, df_pts, self.overlay_png)
                print(f"Saved overlay plot -> {self.overlay_png}")
                
    def analyze_with_progress(self, points: np.ndarray, progress_callback=None):
        """
        Analyze points with progress reporting
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points)
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be Nx3 array")

        results = []
        z_centers = self.calculate_z_centers(points)
        total = len(z_centers)

        for i, zc in enumerate(z_centers):
            mask = (points[:, 2] >= zc - self.dz) & (points[:, 2] <= zc + self.dz)
            slice_points = points[mask]
            
            if len(slice_points) < self.config.min_points_per_slice:
                continue

            try:
                result_tuple, edge_df, warn = self.process_slice(slice_points, zc)
                if result_tuple is not None:
                    result_dict, _, _, _, _ = result_tuple  # Unpack the tuple
                    results.append(result_dict)  # Append only the dictionary
            except Exception as e:
                print(f"Error processing slice at z={zc}: {str(e)}")
                continue

            if progress_callback:
                progress = int((i + 1) / total * 100)
                progress_callback(progress)

        return results  # Now returns list of dictionaries

    def calculate_z_centers(self, points: np.ndarray) -> np.ndarray:
        """Calculate z-centers for slicing"""
        z_min, z_max = points[:,2].min(), points[:,2].max()
        
        if z_max - z_min < self.config.window_len:
            return np.array([(z_min + z_max) / 2.0])
        else:
            return np.arange(z_min + self.dz, 
                            z_max - self.dz + 1e-9, 
                            self.config.z_step)
