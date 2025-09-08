"""
Example usage of cylinder analysis
"""
from analyzer import CylinderAnalyzer, Config  # Modified import

def main():
    # Create configuration
    config = Config(
        window_len = 9.0,
        z_step = 2.0,
        max_points_for_speed = 1_000_000,
        min_points_per_slice = 1000,
        inlier_quantile = 0.80,
        boundary_method = 'angle_max',
        angle_bins = 720,
        draw_per_slice_images = True,
        slice_plot_sample_points = 30000,
        overlay_all = True
    )
    
    # Create analyzer
    analyzer = CylinderAnalyzer(config)
    
    # Process file
    analyzer.analyze_file(r"C:\Users\MAY02\Documents\E3C\NP-05_SCN0001.txt")

if __name__ == "__main__":
    main()
