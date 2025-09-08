import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
                            QTableWidget, QTableWidgetItem, QTabWidget, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from analyzer import CylinderAnalyzer, Config
from utils.data_io import load_txt_points
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import pandas as pd
import os

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, points, params):
        super().__init__()
        self.points = points
        self.params = params
        self.analyzer = CylinderAnalyzer(Config(**params))
        
    def run(self):
        try:
            results = self.analyzer.analyze_with_progress(
                self.points, 
                progress_callback=self.progress.emit
            )
            self.finished.emit({"status": "success", "results": results})
        except Exception as e:
            self.error.emit(str(e))

class LoadPointCloudWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        
    def run(self):
        try:
            # Read file in chunks to show progress
            total_lines = sum(1 for _ in open(self.filename))
            points = []
            
            with open(self.filename) as f:
                for i, line in enumerate(f):
                    # Skip comments and empty lines
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            coords = [float(x) for x in line.replace(',', ' ').split()]
                            if len(coords) >= 3:
                                points.append(coords[:3])
                        except ValueError:
                            continue
                    
                    # Update progress every 1000 lines
                    if i % 1000 == 0:
                        progress = int((i + 1) / total_lines * 100)
                        self.progress.emit(progress)
            
            points = np.array(points)
            self.finished.emit((points, len(points)))
            
        except Exception as e:
            self.error.emit(str(e))

class CylinderAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cylinder Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(300)
        
        # File loading section
        load_btn = QPushButton("Load Point Cloud")
        load_btn.clicked.connect(self.load_file)
        control_layout.addWidget(load_btn)
        
        # Parameters section
        param_group = QWidget()
        param_layout = QVBoxLayout(param_group)
        
        # Z-window parameters
        self.z_window = QDoubleSpinBox()
        self.z_window.setRange(1.0, 100.0)
        self.z_window.setValue(9.0)
        self.z_window.setSingleStep(0.5)
        param_layout.addWidget(QLabel("Window Length (mm):"))
        param_layout.addWidget(self.z_window)
        
        self.z_step = QDoubleSpinBox()
        self.z_step.setRange(0.1, 50.0)
        self.z_step.setValue(2.0)
        self.z_step.setSingleStep(0.1)
        param_layout.addWidget(QLabel("Z Step (mm):"))
        param_layout.addWidget(self.z_step)
        
        # Boundary method selection
        self.boundary_method = QComboBox()
        self.boundary_method.addItems(['angle_max', 'convex_hull'])
        param_layout.addWidget(QLabel("Boundary Method:"))
        param_layout.addWidget(self.boundary_method)
        
        self.angle_bins = QSpinBox()
        self.angle_bins.setRange(90, 3600)
        self.angle_bins.setValue(720)
        self.angle_bins.setSingleStep(90)
        param_layout.addWidget(QLabel("Angle Bins:"))
        param_layout.addWidget(self.angle_bins)
        
        # Analysis button
        analyze_btn = QPushButton("Analyze Slice")
        analyze_btn.clicked.connect(self.analyze_current)
        param_layout.addWidget(analyze_btn)
        
        # Add export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_btn.setEnabled(False)  # Disable until we have results
        self.export_btn = export_btn  # Store reference to enable/disable
        param_layout.addWidget(export_btn)
        
        control_layout.addWidget(param_group)
        control_layout.addStretch()
        
        # Add Z slice visualization controls
        slice_viz_group = QWidget()
        slice_viz_layout = QVBoxLayout(slice_viz_group)
        
        self.z_slice_input = QDoubleSpinBox()
        self.z_slice_input.setRange(-1000, 1000)
        self.z_slice_input.setValue(0.0)
        self.z_slice_input.setSingleStep(1.0)
        slice_viz_layout.addWidget(QLabel("Z Position to Visualize:"))
        slice_viz_layout.addWidget(self.z_slice_input)
        
        show_slice_btn = QPushButton("Show Slice at Z")
        show_slice_btn.clicked.connect(self.visualize_slice)
        slice_viz_layout.addWidget(show_slice_btn)
        
        # Add to control panel after parameters
        control_layout.addWidget(slice_viz_group)
        
        # Compare with other years button
        compare_btn = QPushButton("Compare with Other Years")
        compare_btn.clicked.connect(self.compare_years)
        control_layout.addWidget(compare_btn)
        
        # VTK Widget for 3D visualization
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.renderer.SetBackground(0.1, 0.1, 0.1)
        
        # Add tabs for results
        self.results_tabs = QTabWidget()
        self.table_tab = QWidget()
        self.plots_tab = QWidget()
        self.results_tabs.addTab(self.table_tab, "Results Table")
        self.results_tabs.addTab(self.plots_tab, "Plots")
        
        # Setup table
        self.results_table = QTableWidget()
        table_layout = QVBoxLayout(self.table_tab)
        table_layout.addWidget(self.results_table)
        
        # Setup plots
        plots_layout = QVBoxLayout(self.plots_tab)
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        plots_layout.addWidget(self.canvas)
        
        # Add slice visualization tab
        self.slice_tab = QWidget()
        self.results_tabs.addTab(self.slice_tab, "Slice View")
        slice_tab_layout = QVBoxLayout(self.slice_tab)
        
        # Initialize slice visualization figure
        self.slice_figure = plt.figure(figsize=(8, 8))
        self.slice_canvas = FigureCanvasQTAgg(self.slice_figure)
        slice_tab_layout.addWidget(self.slice_canvas)
        
        # Add widgets to main layout using QSplitter
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create splitter for VTK view and results
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.vtk_widget)
        self.splitter.addWidget(self.results_tabs)
        
        # Set initial sizes (2:1 ratio)
        self.splitter.setSizes([600, 300])
        self.splitter.setStretchFactor(0, 2)  # VTK widget stretches more
        self.splitter.setStretchFactor(1, 1)  # Results tabs stretch less
        
        right_layout.addWidget(self.splitter)
        layout.addWidget(control_panel)
        layout.addWidget(right_panel)
        
        self.points = None
        self.point_cloud_actor = None
        
        # Initialize VTK interaction
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # Add stats display
        self.stats_label = QLabel()
        control_layout.addWidget(self.stats_label)
        
    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Point Cloud",
            "",
            "Point Cloud Files (*.txt *.csv *.xyz);;All Files (*.*)"
        )
        
        if filename:
            # Show progress bar and disable UI
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.setEnabled(False)
            self.statusBar().showMessage("Loading point cloud...")
            
            # Create and start worker thread
            self.load_worker = LoadPointCloudWorker(filename)
            self.load_worker.progress.connect(self.update_progress)
            self.load_worker.finished.connect(self.load_finished)
            self.load_worker.error.connect(self.load_error)
            self.load_worker.start()

    def load_finished(self, result):
        points, num_points = result
        self.points = points
        self.display_point_cloud()
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self.statusBar().showMessage(f"Loaded {num_points} points")

    def load_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self.statusBar().showMessage(f"Error loading file: {error_msg}")
    
    def display_point_cloud(self):
        if self.point_cloud_actor:
            self.renderer.RemoveActor(self.point_cloud_actor)
            
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        for point in self.points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
            
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        
        # Vertex filter
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())
        
        self.point_cloud_actor = vtk.vtkActor()
        self.point_cloud_actor.SetMapper(mapper)
        self.point_cloud_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        self.point_cloud_actor.GetProperty().SetPointSize(3)
        
        self.renderer.AddActor(self.point_cloud_actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
    
    def analyze_current(self):
        if self.points is None:
            self.statusBar().showMessage("Please load point cloud first")
            return
            
        # Get parameters from GUI
        params = {
            'window_len': self.z_window.value(),
            'z_step': self.z_step.value(),
            'boundary_method': self.boundary_method.currentText(),
            'angle_bins': self.angle_bins.value(),
            'draw_per_slice_images': True,
            'slice_plot_sample_points': 30000,
            'overlay_all': True,
            'max_points_for_speed': 1_000_000,
            'min_points_per_slice': 1000,
            'inlier_quantile': 0.80
        }
        
        # Disable UI during analysis
        self.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.worker = AnalysisWorker(self.points, params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.start()

    # Add the missing error handler method
    def analysis_error(self, error_msg):
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage(f"Error during analysis: {error_msg}")
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(f"Processing: {value}%")
    
    def analysis_finished(self, results):
        self.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if results["status"] == "success":
            self.current_results = results["results"]
            self.export_btn.setEnabled(True)
            self.display_results(self.current_results)
            self.show_statistics()  # Add statistics display
            self.statusBar().showMessage("Analysis completed successfully")
    
    def display_results(self, results):
        # Update table with all columns
        headers = [
            "Z Center", "Z Low", "Z High", "Points", "Edge Points",
            "Center X", "Center Y", "Radius", 
            "R Max", "R Min",
            "X at Rmax", "Y at Rmax",
            "X at Rmin", "Y at Rmin",
            "Ovality Abs", "Ovality %",
            "k2 Amplitude", "Cost"
        ]
        
        self.results_table.clear()
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(results))
        
        # Add data to table
        for i, res in enumerate(results):
            columns = [
                f"{res['z_center']:.3f}",
                f"{res['z_low']:.3f}",
                f"{res['z_high']:.3f}",
                str(res['n_points_slice']),
                str(res['n_edge']),
                f"{res['cx']:.3f}",
                f"{res['cy']:.3f}",
                f"{res['R']:.3f}",
                f"{res['Rmax']:.3f}",
                f"{res['Rmin']:.3f}",
                f"{res['x_at_Rmax']:.3f}",
                f"{res['y_at_Rmax']:.3f}",
                f"{res['x_at_Rmin']:.3f}",
                f"{res['y_at_Rmin']:.3f}",
                f"{res['ovality_abs']:.3f}",
                f"{res['ovality_pct']:.3f}",
                f"{res['k2_amp']:.3f}",
                f"{res['cost']:.3f}"
            ]
            
            for j, value in enumerate(columns):
                self.results_table.setItem(i, j, QTableWidgetItem(value))
        
        # Auto-resize columns
        self.results_table.resizeColumnsToContents()

    def show_statistics(self):
        if self.points is None:
            return
            
        z_values = self.points[:, 2]
        stats = {
            'count': len(z_values),
            'mean': np.mean(z_values),
            'median': np.median(z_values),
            'min': np.min(z_values),
            'max': np.max(z_values),
            'std': np.std(z_values),
            'range': np.max(z_values) - np.min(z_values)
        }
        
        # Create statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self.stats_table.setRowCount(len(stats))
        
        # Add statistics to table
        for i, (key, value) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(key.capitalize()))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{value:.3f}"))
        
        # Add to a new tab
        if not hasattr(self, 'stats_tab'):
            self.stats_tab = QWidget()
            self.results_tabs.addTab(self.stats_tab, "Statistics")
            stats_layout = QVBoxLayout(self.stats_tab)
            
            # Add description label
            desc_label = QLabel("Statistics of Z-axis measurements:")
            desc_label.setStyleSheet("font-weight: bold;")
            stats_layout.addWidget(desc_label)
            stats_layout.addWidget(self.stats_table)
        
        self.stats_table.resizeColumnsToContents()
        
        # Show statistics in status bar
        self.statusBar().showMessage(
            f"Mean Z: {stats['mean']:.3f}, Range: {stats['range']:.3f}, "
            f"Std Dev: {stats['std']:.3f}"
        )

    # Add this method to the CylinderAnalyzerGUI class

    def load_comparison_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point Cloud Files for Comparison",
            "",
            "Point Cloud Files (*.txt *.csv *.xyz);;All Files (*.*)"
        )
        
        if not filenames:
            return []
            
        comparison_data = []
        for filename in filenames:
            try:
                # Show progress
                self.statusBar().showMessage(f"Loading {filename}...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # Create and run worker for each file
                worker = LoadPointCloudWorker(filename)
                worker.progress.connect(self.update_progress)
                worker.start()
                worker.wait()  # Wait for completion
                
                # Get year from filename (assuming format YYYY_*.txt)
                try:
                    year = os.path.basename(filename).split('_')[0]
                except:
                    year = os.path.basename(filename)
                
                points = np.loadtxt(filename)
                if points.shape[1] >= 3:  # Ensure we have x,y,z columns
                    comparison_data.append((year, points))
                
            except Exception as e:
                self.statusBar().showMessage(f"Error loading {filename}: {str(e)}")
                continue
                
        self.progress_bar.setVisible(False)
        return comparison_data

    # Add this method to CylinderAnalyzerGUI class
    def export_results(self):
        if not hasattr(self, 'current_results'):
            self.statusBar().showMessage("No results to export")
            return
            
        # Get directory to save files
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Save Results",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not save_dir:
            return
            
        try:
            # Get current year
            from datetime import datetime
            current_year = datetime.now().year
            
            # Create results DataFrame
            df_results = pd.DataFrame(self.current_results)
            df_results['Year'] = current_year  # Add year column
            
            # Save main results with year in filename
            results_path = os.path.join(save_dir, f"{current_year}_cylinder_analysis_results.csv")
            df_results.to_csv(results_path, index=False, float_format='%.6f')
            
            # Save slice statistics
            stats_path = os.path.join(save_dir, "slice_statistics.csv")
            z_values = self.points[:, 2]
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                'Value': [
                    np.mean(z_values),
                    np.median(z_values),
                    np.min(z_values),
                    np.max(z_values),
                    np.std(z_values)
                ]
            })
            stats_df.to_csv(stats_path, index=False)
            
            # Save current slice data if available
            if hasattr(self, 'current_slice_data'):
                slice_path = os.path.join(save_dir, "current_slice_data.csv")
                slice_df = pd.DataFrame(self.current_slice_data)
                slice_df.to_csv(slice_path, index=False, float_format='%.6f')
            
            self.statusBar().showMessage(
                f"Results exported to {save_dir}"
            )
            
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting results: {str(e)}")
    
    # Add this method to CylinderAnalyzerGUI class

    def visualize_slice(self):
        if self.points is None:
            self.statusBar().showMessage("Please load point cloud first")
            return
            
        z_target = self.z_slice_input.value()
        window_len = self.z_window.value()
        dz = window_len / 2.0
        
        # Get slice points
        mask = (self.points[:, 2] >= z_target - dz) & (self.points[:, 2] <= z_target + dz)
        slice_points = self.points[mask]
        
        if len(slice_points) < 1000:
            self.statusBar().showMessage(f"Too few points ({len(slice_points)}) in slice")
            return
            
        # Configure parameters
        params = {
            'window_len': window_len,
            'z_step': self.z_step.value(),
            'boundary_method': self.boundary_method.currentText(),
            'angle_bins': self.angle_bins.value(),
            'max_points_for_speed': 1_000_000,
            'min_points_per_slice': 1000,
            'inlier_quantile': 0.80
        }
        
        try:
            # Process single slice
            analyzer = CylinderAnalyzer(Config(**params))
            result_tuple = analyzer.process_slice(slice_points, z_target)
            
            if result_tuple is None:
                self.statusBar().showMessage("Could not process slice")
                return
                
            result_dict, edge_xy, _, (xc, yc, R), ov = result_tuple
            
            # Clear previous plot
            self.slice_figure.clear()
            ax = self.slice_figure.add_subplot(111)
            
            # Plot points (subsample if too many)
            if len(slice_points) > 30000:
                idx = np.random.choice(len(slice_points), 30000, replace=False)
                plot_points = slice_points[idx]
            else:
                plot_points = slice_points
            
            # Plot points, boundary and fitted circle
            ax.scatter(plot_points[:, 0], plot_points[:, 1], 
                      s=1, alpha=0.3, label='Points')
            ax.plot(edge_xy[:, 0], edge_xy[:, 1], 'k-', 
                   linewidth=1, label='Boundary')
            
            # Plot fitted circle
            theta = np.linspace(0, 2*np.pi, 360)
            circle_x = xc + R*np.cos(theta)
            circle_y = yc + R*np.sin(theta)
            ax.plot(circle_x, circle_y, 'r-', 
                   linewidth=2, label='Fitted Circle')
            
            # Plot center
            ax.plot(xc, yc, 'r+', markersize=10, label='Center')
            
            # Add text with parameters
            info_text = (
                f"Z = {z_target:.3f} ± {dz:.1f}\n"
                f"Points: {len(slice_points)}\n"
                f"Center: ({xc:.3f}, {yc:.3f})\n"
                f"Radius: {R:.3f}\n"
                f"Ovality: {result_dict['ovality_pct']:.3f}%"
            )
            ax.text(0.02, 0.98, info_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_aspect('equal')
            ax.grid(True)
            
            # Adjust layout before adding legend
            self.slice_figure.tight_layout()
            
            # Add legend with better placement
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust subplot parameters to fit legend
            self.slice_figure.subplots_adjust(right=0.85)
            
            self.slice_canvas.draw()
            
            # Switch to slice tab
            self.results_tabs.setCurrentIndex(2)
            
            self.statusBar().showMessage("Slice visualization complete")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error visualizing slice: {str(e)}")

    def compare_years(self):
        # Select CSV files for comparison
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Results CSV Files for Comparison",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if not filenames:
            return
            
        try:
            # Load and combine data
            all_data = []
            for filename in filenames:
                # Extract year from filename
                year = os.path.basename(filename).split('_')[0]
                
                # Load CSV
                df = pd.read_csv(filename)
                df['Year'] = year  # Add year column
                all_data.append(df)
            
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Create comparison plots
            self.figure.clear()
            
            # Plot 1: Ovality vs Z for different years
            ax1 = self.figure.add_subplot(211)
            for year in combined_df['Year'].unique():
                year_data = combined_df[combined_df['Year'] == year]
                ax1.plot(year_data['z_center'], year_data['ovality_pct'], 
                        '-o', label=f'Year {year}', markersize=4)
            
            ax1.set_xlabel('Z Position')
            ax1.set_ylabel('Ovality %')
            ax1.grid(True)
            ax1.legend()
            ax1.set_title('Ovality Comparison')
            
            # Plot 2: Radius vs Z for different years
            ax2 = self.figure.add_subplot(212)
            for year in combined_df['Year'].unique():
                year_data = combined_df[combined_df['Year'] == year]
                ax2.plot(year_data['z_center'], year_data['R'], 
                        '-o', label=f'Year {year}', markersize=4)
            
            ax2.set_xlabel('Z Position')
            ax2.set_ylabel('Radius')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Radius Comparison')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Switch to plots tab
            self.results_tabs.setCurrentIndex(1)
            
            # Create comparison statistics
            stats_text = "Comparison Statistics:\n\n"
            for year in combined_df['Year'].unique():
                year_data = combined_df[combined_df['Year'] == year]
                stats_text += f"Year {year}:\n"
                stats_text += f"  Mean Ovality: {year_data['ovality_pct'].mean():.3f}%\n"
                stats_text += f"  Max Ovality: {year_data['ovality_pct'].max():.3f}%\n"
                stats_text += f"  Mean Radius: {year_data['R'].mean():.3f}\n"
                stats_text += f"  Radius Range: {year_data['R'].max() - year_data['R'].min():.3f}\n\n"
            
            # Update stats label
            self.stats_label.setText(stats_text)
            
            self.statusBar().showMessage("Comparison completed successfully")
            
        except Exception as e:
            self.statusBar().showMessage(f"Error comparing data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CylinderAnalyzerGUI()
    window.show()
    sys.exit(app.exec())