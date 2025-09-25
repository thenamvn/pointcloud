import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
                            QTableWidget, QTableWidgetItem, QTabWidget, QSplitter, QCheckBox, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from analyzer import CylinderAnalyzer, Config
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import pandas as pd
import os
from datetime import datetime

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

class LoadMultiplePointCloudWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(tuple)
    error = pyqtSignal(str)
    
    def __init__(self, filenames):
        super().__init__()
        self.filenames = filenames
        
    def run(self):
        try:
            all_points = []
            total_files = len(self.filenames)
            
            for file_idx, filename in enumerate(self.filenames):
                # Calculate base progress for this file
                base_progress = int((file_idx / total_files) * 90)
                file_progress_range = int(90 / total_files)  # Each file gets portion of 90%
                
                try:
                    # Parse each file with progress tracking
                    file_points = self.parse_file_with_progress(filename, base_progress, file_progress_range)
                    
                    if len(file_points) == 0:
                        print(f"Warning: No valid points found in {filename}")
                        continue
                        
                    all_points.extend(file_points)
                    print(f"Loaded {len(file_points):,} points from {os.path.basename(filename)}")
                    
                except Exception as e:
                    self.error.emit(f"Error reading {filename}: {str(e)}")
                    return
                
                # Update progress after each file
                self.progress.emit(int(((file_idx + 1) / total_files) * 90))
            
            self.progress.emit(95)
            
            if len(all_points) == 0:
                self.error.emit("No points loaded from any file")
                return
                
            # Convert to numpy array
            points = np.array(all_points, dtype=np.float64)
            
            # Validate and clean data
            if points.shape[1] < 3:
                self.error.emit("Points must have at least 3 coordinates (X, Y, Z)")
                return
                
            # Take only first 3 columns and remove invalid points
            points = points[:, :3]
            valid_mask = np.all(np.isfinite(points), axis=1)
            points = points[valid_mask]
            
            if len(points) == 0:
                self.error.emit("No valid points after filtering")
                return
            
            self.progress.emit(100)
            self.finished.emit((points, len(points)))
            
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
    
    def parse_file_with_progress(self, filename, base_progress, progress_range):
        """Parse single file with progress updates"""
        points = []
        
        try:
            # Try pandas for CSV files first
            if filename.lower().endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(filename)
                if len(df.columns) >= 3:
                    return df.iloc[:, :3].values.tolist()
        except:
            pass
        
        # Manual parsing for text files with progress
        try:
            # Count total lines first for progress calculation
            with open(filename, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            # Parse with progress every 5000 lines to avoid too frequent updates
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#') or line.startswith('//'):
                        continue
                    
                    try:
                        # Handle different separators
                        line = line.replace(',', ' ').replace(';', ' ').replace('\t', ' ')
                        coords = [float(x) for x in line.split() if x]
                        
                        if len(coords) >= 3:
                            points.append(coords[:3])
                            
                    except ValueError:
                        continue
                    
                    # Update progress every 5000 lines to avoid too frequent updates
                    if line_num % 5000 == 0 and total_lines > 0:
                        file_progress = int((line_num / total_lines) * progress_range)
                        self.progress.emit(base_progress + file_progress)
        
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            
        return points
    
class CylinderAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cylinder Analyzer")
        self.setMinimumSize(1400, 800)
        
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

        self.z_min_input = QDoubleSpinBox()
        self.z_min_input.setRange(-10000, 10000)  # Wide range
        self.z_min_input.setValue(0.0)  # Default, will be updated on data load
        self.z_min_input.setSingleStep(1.0)
        self.z_min_input.setDecimals(3)
        self.z_min_input.setEnabled(False)  # Disable until data loads
        param_layout.addWidget(QLabel("Z Min (m):"))
        param_layout.addWidget(self.z_min_input)

        self.z_max_input = QDoubleSpinBox()
        self.z_max_input.setRange(-10000, 10000)
        self.z_max_input.setValue(0.0)  # Default, will be updated on data load
        self.z_max_input.setSingleStep(1.0)
        self.z_max_input.setDecimals(3)
        self.z_max_input.setEnabled(False)
        param_layout.addWidget(QLabel("Z Max (m):"))
        param_layout.addWidget(self.z_max_input)

        # Z-window parameters
        self.z_window = QDoubleSpinBox()
        # self.z_window.setRange(0.001, 0.1)  # 1mm to 100mm in meters
        self.z_window.setValue(9)  # 9 in meters
        self.z_window.setSingleStep(0.5)  # 0.5mm in meters
        self.z_window.setDecimals(4)  # Show 4 decimal places for meters
        self.z_window.setEnabled(True)
        param_layout.addWidget(QLabel("Window Length (m):"))
        param_layout.addWidget(self.z_window)
        
        self.z_step = QDoubleSpinBox()
        # self.z_step.setRange(0.0001, 0.05)  # 0.1mm to 50mm in meters
        self.z_step.setValue(2)  # 2 in meters
        self.z_step.setSingleStep(0.1)  # 0.1mm in meters
        self.z_step.setDecimals(4)  # Show 4 decimal places for meters
        self.z_step.setEnabled(True)
        self.z_step.setReadOnly(False)
        param_layout.addWidget(QLabel("Z Step (m):"))
        param_layout.addWidget(self.z_step)
        
        # Add slice thickness control
        self.slice_thickness = QDoubleSpinBox()
        # self.slice_thickness.setRange(0.001, 0.05)  # 1mm to 50mm in meters
        self.slice_thickness.setValue(0.5)  # 5mm default
        self.slice_thickness.setSingleStep(0.1)
        self.slice_thickness.setDecimals(4)
        self.slice_thickness.setEnabled(True)
        
        param_layout.addWidget(QLabel("Slice Thickness (m):"))
        param_layout.addWidget(self.slice_thickness)
        
        # Boundary method selection
        self.boundary_method = QComboBox()
        self.boundary_method.addItems(['angle_max', 'convex_hull'])
        param_layout.addWidget(QLabel("Boundary Method:"))
        param_layout.addWidget(self.boundary_method)
        
        self.angle_bins = QSpinBox()
        self.angle_bins.setRange(90, 3600)
        self.angle_bins.setValue(720)
        self.angle_bins.setSingleStep(90)
        self.angle_bins.setEnabled(True)
        param_layout.addWidget(QLabel("Angle Bins:"))
        param_layout.addWidget(self.angle_bins)

        quality_group = QWidget()
        quality_layout = QVBoxLayout(quality_group)  # Changed to vertical
        
        quality_layout.addWidget(QLabel("Display Quality:"))
        
        self.quality_dropdown = QComboBox()
        self.quality_dropdown.addItems(["Ultra High", "High", "Medium", "Fast"])
        self.quality_dropdown.setCurrentText("Medium")  # Default
        self.quality_dropdown.currentTextChanged.connect(self.on_quality_changed)
        quality_layout.addWidget(self.quality_dropdown)
        
        control_layout.addWidget(quality_group)
        self.display_quality = 'medium'

        # Add checkbox for auto visualization
        self.auto_visualize = QCheckBox("Auto visualize slices after analysis")
        self.auto_visualize.setChecked(True)  # Default to enabled
        param_layout.addWidget(self.auto_visualize)
    
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
        
        # NEW: Add export data year button
        export_year_btn = QPushButton("Export Data Year")
        export_year_btn.clicked.connect(self.export_data_year)
        export_year_btn.setEnabled(False)  # Disable until we have results
        self.export_year_btn = export_year_btn  # Store reference to enable/disable
        param_layout.addWidget(export_year_btn)
        
        control_layout.addWidget(param_group)
        control_layout.addStretch()

        # Add memory monitor and controls
        memory_group = QWidget()
        memory_layout = QVBoxLayout(memory_group)

        # Memory display
        self.memory_label = QLabel("Memory: 0 MB")
        memory_layout.addWidget(self.memory_label)

        # Add clear data button
        clear_btn = QPushButton("Clear All Data")
        clear_btn.clicked.connect(self.clear_all_data)
        clear_btn.setStyleSheet("QPushButton { color: red; }")
        memory_layout.addWidget(clear_btn)

        control_layout.addWidget(memory_group)

        # Memory monitoring timer
        from PyQt6.QtCore import QTimer
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_display)
        self.memory_timer.start(3000)  # Update every 3 seconds

        # Add Z slice visualization controls with range limits
        slice_viz_group = QWidget()
        slice_viz_layout = QVBoxLayout(slice_viz_group)
        
        self.z_slice_input = QDoubleSpinBox()
        # Initial range (will be updated when data loads)
        self.z_slice_input.setRange(-1000, 1000)
        self.z_slice_input.setValue(0.0)
        self.z_slice_input.setSingleStep(1.0)
        self.z_slice_input.setEnabled(False)  # Disable until data loads
        
        slice_viz_layout.addWidget(QLabel("Z Position to Visualize:"))
        slice_viz_layout.addWidget(self.z_slice_input)

        # Add slice thickness input for visualization (separate from analysis thickness)
        self.viz_slice_thickness = QDoubleSpinBox()
        self.viz_slice_thickness.setRange(0.1, 10.0)  # 0.1m to 10m
        self.viz_slice_thickness.setValue(2.0)  # 2m default (larger than analysis thickness)
        self.viz_slice_thickness.setSingleStep(0.1)
        self.viz_slice_thickness.setDecimals(3)
        self.viz_slice_thickness.setEnabled(False)  # Disable until data loads
        slice_viz_layout.addWidget(QLabel("Viz Slice Thickness (m):"))
        slice_viz_layout.addWidget(self.viz_slice_thickness)

        show_slice_btn = QPushButton("Show Slice at Z")
        show_slice_btn.clicked.connect(self.visualize_slice)
        show_slice_btn.setEnabled(False)  # Disable until data loads
        self.show_slice_btn = show_slice_btn
        slice_viz_layout.addWidget(show_slice_btn)

        # NEW: Add export slice button
        export_slice_btn = QPushButton("Export Slice")
        export_slice_btn.clicked.connect(self.export_slice)
        export_slice_btn.setEnabled(False)  # Disable until slice is visualized
        self.export_slice_btn = export_slice_btn
        slice_viz_layout.addWidget(export_slice_btn)
        
        # Add to control panel after parameters
        control_layout.addWidget(slice_viz_group)
        
        # Compare with other years button
        compare_btn = QPushButton("Compare with Other Years")
        compare_btn.clicked.connect(self.compare_years)
        control_layout.addWidget(compare_btn)
        
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
        
        # Create splitter for VTK views and results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create horizontal splitter for two VTK views
        vtk_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left VTK widget for original point cloud
        self.vtk_widget_original = QVTKRenderWindowInteractor()
        self.renderer_original = vtk.vtkRenderer()
        self.vtk_widget_original.GetRenderWindow().AddRenderer(self.renderer_original)
        self.renderer_original.SetBackground(0.1, 0.1, 0.1)
        
        # Right VTK widget for slice visualization
        self.vtk_widget_slices = QVTKRenderWindowInteractor()
        self.renderer_slices = vtk.vtkRenderer()
        self.vtk_widget_slices.GetRenderWindow().AddRenderer(self.renderer_slices)
        self.renderer_slices.SetBackground(0.1, 0.2, 0.2)
        
        # Add labels for each view
        original_frame = QWidget()
        original_layout = QVBoxLayout(original_frame)
        original_layout.addWidget(QLabel("Original Point Cloud"))
        original_layout.addWidget(self.vtk_widget_original)
        
        slices_frame = QWidget()
        slices_layout = QVBoxLayout(slices_frame)
        slices_layout.addWidget(QLabel("Reconstructed Slices"))
        slices_layout.addWidget(self.vtk_widget_slices)
        
        vtk_splitter.addWidget(original_frame)
        vtk_splitter.addWidget(slices_frame)
        vtk_splitter.setSizes([700, 700])  # Equal sizes
        
        # Create main vertical splitter
        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(vtk_splitter)
        self.main_splitter.addWidget(self.results_tabs)
        
        # Set initial sizes (2:1 ratio)
        self.main_splitter.setSizes([800, 400])
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 1)
        
        right_layout.addWidget(self.main_splitter)
        layout.addWidget(control_panel)
        layout.addWidget(right_panel)
        
        self.points = None
        self.point_cloud_actor = None
        
        # Initialize VTK interactions
        self.iren_original = self.vtk_widget_original.GetRenderWindow().GetInteractor()
        self.iren_slices = self.vtk_widget_slices.GetRenderWindow().GetInteractor()
        self.iren_original.Initialize()
        self.iren_slices.Initialize()
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        control_layout.addWidget(self.progress_bar)
        
        # Add stats display
        self.stats_label = QLabel()
        control_layout.addWidget(self.stats_label)

    def on_quality_changed(self, quality_text):
        """Handle quality dropdown change"""
        quality_map = {
            "Ultra High": "ultra_high",
            "High": "high", 
            "Medium": "medium",
            "Fast": "fast"
        }
        
        self.display_quality = quality_map[quality_text]
        
        if hasattr(self, 'points') and self.points is not None:
            self.display_point_cloud()
            
            # Show quality info
            total_points = len(self.points)
            display_points = len(self.get_display_points())
            ratio = (display_points / total_points) * 100
            
            self.statusBar().showMessage(
                f"Display quality: {quality_text} - Showing {display_points:,}/{total_points:,} points ({ratio:.1f}%)"
            )


    def update_memory_display(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            
            if hasattr(self, 'points') and self.points is not None:
                points_mb = self.points.nbytes / (1024*1024)
                self.memory_label.setText(f"RAM: {memory_mb:.0f}MB (Points: {points_mb:.0f}MB)")
                
                # Color coding
                if memory_mb > 6000:  # 6GB - red
                    self.memory_label.setStyleSheet("color: red; font-weight: bold;")
                elif memory_mb > 4000:  # 4GB - orange
                    self.memory_label.setStyleSheet("color: orange; font-weight: bold;")
                elif memory_mb > 2000:  # 2GB - yellow
                    self.memory_label.setStyleSheet("color: #FF8C00; font-weight: bold;")
                else:  # < 2GB - green
                    self.memory_label.setStyleSheet("color: green;")
            else:
                self.memory_label.setText(f"RAM: {memory_mb:.0f}MB (No data)")
                self.memory_label.setStyleSheet("color: gray;")
                
        except ImportError:
            self.memory_label.setText("RAM: psutil not available")
        except Exception:
            pass

    def clear_all_data(self):
        """Clear all loaded data to free memory"""
        from PyQt6.QtWidgets import QMessageBox
        
        if not hasattr(self, 'points') or self.points is None:
            self.statusBar().showMessage("No data to clear")
            return
        
        reply = QMessageBox.question(
            self,
            "Clear All Data",
            f"Are you sure you want to clear {len(self.points):,} points?\n"
            f"This will free up memory but you'll lose all current data.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Clear all data
            if hasattr(self, 'points'):
                del self.points
                self.points = None
            
            if hasattr(self, 'current_results'):
                del self.current_results
            
            # Clear VTK actors
            if hasattr(self, 'point_cloud_actor') and self.point_cloud_actor:
                self.renderer_original.RemoveActor(self.point_cloud_actor)
                self.point_cloud_actor = None
            
            if hasattr(self, 'slice_actors'):
                for actor in self.slice_actors:
                    self.renderer_slices.RemoveActor(actor)
                self.slice_actors = []
            
            # Clear plots
            self.figure.clear()
            self.canvas.draw()
            self.slice_figure.clear()
            self.slice_canvas.draw()
            
            # Update VTK
            self.vtk_widget_original.GetRenderWindow().Render()
            self.vtk_widget_slices.GetRenderWindow().Render()
            
            # Disable controls
            self.z_slice_input.setEnabled(False)
            self.show_slice_btn.setEnabled(False)
            self.viz_slice_thickness.setEnabled(False)
            self.export_btn.setEnabled(False)
            self.export_year_btn.setEnabled(False)  # NEW: Disable export year button
            self.export_slice_btn.setEnabled(False)  # NEW: Disable export slice button
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.statusBar().showMessage("All data cleared - memory freed")

    def load_file(self):
        # Allow multi-file selection but load incrementally
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point Cloud File(s) - Multiple files will be loaded incrementally",
            "",
            "Point Cloud Files (*.txt *.csv *.xyz);;All Files (*.*)"
        )
        
        if filenames:
            # Auto-detect based on number of files selected
            if len(filenames) == 1:
                self.statusBar().showMessage(f"Loading single file: {os.path.basename(filenames[0])}...")
            else:
                self.statusBar().showMessage(f"Loading and combining {len(filenames)} files...")
            
            self.start_incremental_loading(filenames)

    def start_incremental_loading(self, filenames):
        """Load multiple files incrementally to avoid memory overflow"""
        self.files_to_load = filenames.copy()
        self.total_files = len(filenames)
        self.current_file_index = 0
        self.files_loaded = 0
        
        # Show progress bar and disable UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.setEnabled(False)
        
        # Start loading first file
        self.load_next_file()

    def load_next_file(self):
        """Load the next file in the queue"""
        if self.current_file_index >= len(self.files_to_load):
            # All files loaded, finish up
            self.finish_incremental_loading()
            return
        
        current_file = self.files_to_load[self.current_file_index]
        
        # Calculate overall progress
        overall_progress = int((self.current_file_index / self.total_files) * 90)
        self.progress_bar.setValue(overall_progress)
        
        # Show status
        remaining = self.total_files - self.current_file_index
        self.statusBar().showMessage(
            f"Loading file {self.current_file_index + 1}/{self.total_files}: "
            f"{os.path.basename(current_file)} ({remaining} remaining)"
        )
        
        # Create worker for current file
        self.current_load_worker = LoadPointCloudWorker(current_file)
        self.current_load_worker.progress.connect(self.update_file_progress)
        self.current_load_worker.finished.connect(self.single_file_loaded)
        self.current_load_worker.error.connect(self.incremental_load_error)
        self.current_load_worker.start()

    def update_file_progress(self, file_progress):
        """Update progress for current file loading"""
        # Combine overall progress with current file progress
        overall_progress = int((self.current_file_index / self.total_files) * 90)
        file_contribution = int((file_progress / 100) * (90 / self.total_files))
        total_progress = overall_progress + file_contribution
        self.progress_bar.setValue(min(total_progress, 95))

    def single_file_loaded(self, result):
        """Handle completion of single file loading"""
        new_points, num_new_points = result
        
        try:
            # Get file info for reporting
            current_file = self.files_to_load[self.current_file_index]
            file_memory_mb = new_points.nbytes / (1024*1024)
            
            # Combine with existing data if we have any
            if hasattr(self, 'points') and self.points is not None:
                # Memory check before combining
                existing_memory_mb = self.points.nbytes / (1024*1024)
                total_memory_mb = existing_memory_mb + file_memory_mb
                
                # Warning if getting close to memory limit
                if total_memory_mb > 6000:  # 6GB warning
                    from PyQt6.QtWidgets import QMessageBox
                    
                    remaining_files = self.total_files - self.current_file_index - 1
                    estimated_total_mb = total_memory_mb * (1 + remaining_files * 0.5)  # Rough estimate
                    
                    reply = QMessageBox.warning(
                        self,
                        "High Memory Usage Warning",
                        f"Current memory usage: {total_memory_mb:.0f}MB\n"
                        f"Estimated final usage: {estimated_total_mb:.0f}MB\n"
                        f"Remaining files: {remaining_files}\n\n"
                        f"Continue loading? (You can stop now to avoid memory issues)",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )
                    
                    if reply == QMessageBox.StandardButton.No:
                        # Stop loading, finish with current data
                        self.finish_incremental_loading()
                        return
                
                # Combine data
                old_count = len(self.points)
                combined_points = np.vstack([self.points, new_points])
                
                # Clear old data to free memory
                del self.points
                del new_points
                import gc
                gc.collect()
                
                self.points = combined_points
                
                self.statusBar().showMessage(
                    f"Added {num_new_points:,} points from {os.path.basename(current_file)}. "
                    f"Total: {len(self.points):,} points"
                )
                
            else:
                # First file
                self.points = new_points
                self.statusBar().showMessage(f"Loaded {num_new_points:,} points from {os.path.basename(current_file)}")
            
            self.files_loaded += 1
            
        except MemoryError:
            self.statusBar().showMessage(f"Memory error loading {os.path.basename(current_file)} - stopping here")
            self.finish_incremental_loading()
            return
        except Exception as e:
            self.statusBar().showMessage(f"Error combining {os.path.basename(current_file)}: {str(e)}")
        
        # Move to next file
        self.current_file_index += 1
        
        # Small delay to allow UI updates and memory cleanup
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.load_next_file)

    def incremental_load_error(self, error_msg):
        """Handle error during incremental loading"""
        current_file = self.files_to_load[self.current_file_index]
        self.statusBar().showMessage(f"Error loading {os.path.basename(current_file)}: {error_msg}")
        
        # Skip this file and continue with next
        self.current_file_index += 1
        
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.load_next_file)

    def finish_incremental_loading(self):
        """Finish incremental loading process"""
        # Clean up loading state
        if hasattr(self, 'current_load_worker'):
            del self.current_load_worker
        
        # Update UI if we have data
        if hasattr(self, 'points') and self.points is not None:
            self.update_ui_after_load()
            
            # Show final stats
            final_memory_mb = self.points.nbytes / (1024*1024)
            self.statusBar().showMessage(
                f"Incremental loading complete! Loaded {self.files_loaded}/{self.total_files} files. "
                f"Total: {len(self.points):,} points ({final_memory_mb:.0f}MB)"
            )
        else:
            self.statusBar().showMessage("No data loaded")
            self.setEnabled(True)
            self.progress_bar.setVisible(False)
        
        # Clean up
        if hasattr(self, 'files_to_load'):
            del self.files_to_load

    def load_finished(self, result):
        """Backup method for single file loading (kept for compatibility)"""
        new_points, num_new_points = result
        
        # Handle append vs replace mode (same logic as before)
        if hasattr(self, 'append_mode') and self.append_mode and hasattr(self, 'points') and self.points is not None:
            # Append mode logic (same as before)
            old_count = len(self.points)
            old_memory_mb = self.points.nbytes / (1024*1024)
            new_memory_mb = new_points.nbytes / (1024*1024)
            total_memory_mb = old_memory_mb + new_memory_mb
            
            if total_memory_mb > 4000:  # 4GB warning
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.warning(
                    self,
                    "Memory Warning",
                    f"Combined data will use ~{total_memory_mb:.0f}MB RAM\n"
                    f"Current: {old_memory_mb:.0f}MB + New: {new_memory_mb:.0f}MB\n\n"
                    f"This may cause performance issues. Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.No:
                    self.progress_bar.setVisible(False)
                    self.setEnabled(True)
                    self.statusBar().showMessage("Loading cancelled due to memory concerns")
                    return
            try:
                combined_points = np.vstack([self.points, new_points])
                del self.points
                del new_points
                import gc
                gc.collect()
                
                self.points = combined_points
                total_points = len(self.points)
                
                self.statusBar().showMessage(f"Appended {num_new_points:,} points. Total: {total_points:,} points (was {old_count:,})")
                
            except MemoryError:
                self.statusBar().showMessage("Not enough memory to combine point clouds")
                self.progress_bar.setVisible(False)
                self.setEnabled(True)
                return
            except Exception as e:
                self.statusBar().showMessage(f"Error combining data: {str(e)}")
                self.progress_bar.setVisible(False)
                self.setEnabled(True)
                return
        else:
            # Replace mode
            if hasattr(self, 'points') and self.points is not None:
                del self.points
                import gc
                gc.collect()
            
            self.points = new_points
            self.statusBar().showMessage(f"Loaded {num_new_points:,} points")
        
        # Update UI
        self.update_ui_after_load()
        
        final_memory_mb = self.points.nbytes / (1024*1024)
        self.statusBar().showMessage(f"Ready - {len(self.points):,} points ({final_memory_mb:.0f}MB)")

    def update_ui_after_load(self):
        """Update UI elements after loading data"""
        # Clear old analysis results
        if hasattr(self, 'current_results'):
            del self.current_results
            self.export_btn.setEnabled(False)
            self.export_year_btn.setEnabled(False)  # NEW: Disable export year button
        
        # Clear VTK actors
        if hasattr(self, 'point_cloud_actor') and self.point_cloud_actor:
            self.renderer_original.RemoveActor(self.point_cloud_actor)
            self.point_cloud_actor = None
        
        if hasattr(self, 'slice_actors'):
            for actor in self.slice_actors:
                self.renderer_slices.RemoveActor(actor)
            self.slice_actors = []
        
        # Update Z range based on loaded data
        z_min = np.min(self.points[:, 2])
        z_max = np.max(self.points[:, 2])
        self.z_min_input.setRange(z_min, z_max)
        self.z_min_input.setValue(z_min)
        self.z_max_input.setValue(z_max)
        self.z_max_input.setRange(z_min, z_max)
        self.z_min_input.setEnabled(True)
        self.z_max_input.setEnabled(True)
        z_range = z_max - z_min
        
        # Set Z spinbox range and initial value
        self.z_slice_input.setRange(z_min, z_max)
        self.z_slice_input.setValue(z_min + z_range/2)
        self.z_slice_input.setSingleStep(z_range/50)

        # Set reasonable default thickness based on data range
        default_thickness = max(0.5, z_range / 50)
        self.viz_slice_thickness.setValue(default_thickness)

        # Enable controls
        self.z_slice_input.setEnabled(True)
        self.show_slice_btn.setEnabled(True)
        self.viz_slice_thickness.setEnabled(True)
        
        # Update displays
        self.display_point_cloud()
        self.show_statistics()
        
        # Clear plots
        self.figure.clear()
        self.canvas.draw()
        self.slice_figure.clear()
        self.slice_canvas.draw()
        
        self.progress_bar.setVisible(False)
        self.setEnabled(True)

    def load_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.setEnabled(True)
        self.statusBar().showMessage(f"Error loading file: {error_msg}")

    def set_display_quality(self, quality):
        """Set display quality and refresh if we have data"""
        self.display_quality = quality
        if hasattr(self, 'points') and self.points is not None:
            self.display_point_cloud()

    def display_point_cloud(self):
        if self.point_cloud_actor:
            self.renderer_original.RemoveActor(self.point_cloud_actor)
            
        # Get points based on current quality setting
        display_points = self.get_display_points()
        
        # Get point size from quality setting
        quality_settings = {
            'ultra_high': {'max_points': 2000000, 'point_size': 3},
            'high': {'max_points': 1000000, 'point_size': 3},
            'medium': {'max_points': 200000, 'point_size': 2},
            'fast': {'max_points': 50000, 'point_size': 1}
        }
        point_size = quality_settings[self.display_quality]['point_size']
        
        # Show loading message for large datasets
        if len(display_points) > 500000:
            self.statusBar().showMessage(f"Rendering {len(display_points):,} points (this may take a moment)...")
        
        # Create VTK points
        vtk_points = vtk.vtkPoints()
        for point in display_points:
            vtk_points.InsertNextPoint(point[0], point[1], point[2])
            
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())
        
        self.point_cloud_actor = vtk.vtkActor()
        self.point_cloud_actor.SetMapper(mapper)
        self.point_cloud_actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        self.point_cloud_actor.GetProperty().SetPointSize(point_size)
        
        self.renderer_original.AddActor(self.point_cloud_actor)
        self.renderer_original.ResetCamera()
        self.vtk_widget_original.GetRenderWindow().Render()
        
        # Show completion message with details
        total_points = len(self.points)
        display_ratio = (len(display_points) / total_points) * 100
        quality_name = self.quality_dropdown.currentText()
        
        self.statusBar().showMessage(
            f"{quality_name} quality: {len(display_points):,}/{total_points:,} points ({display_ratio:.1f}%) rendered"
        )
    
    def get_display_points(self):
        """Get subsampled points based on current quality setting"""
        if self.points is None:
            return np.array([])
        
        # Updated quality settings with Ultra High option
        quality_settings = {
            'ultra_high': {'max_points': 2000000, 'point_size': 3},  # 2M points - much higher
            'high': {'max_points': 1000000, 'point_size': 3},        # 1M points
            'medium': {'max_points': 200000, 'point_size': 2},       # 200K points  
            'fast': {'max_points': 50000, 'point_size': 1}          # 50K points
        }
        
        settings = quality_settings.get(self.display_quality, quality_settings['medium'])
        max_points = settings['max_points']
        total_points = len(self.points)
        
        if total_points <= max_points:
            # If data is smaller than limit, show all points
            return self.points
        
        # Smart subsampling based on ratio
        sample_ratio = max_points / total_points
        
        if sample_ratio > 0.8:  # > 80% - use random sampling
            indices = np.random.choice(total_points, max_points, replace=False)
            return self.points[indices]
        elif sample_ratio > 0.5:  # 50-80% - mixed sampling for better coverage
            # Use systematic + random sampling
            systematic_count = int(max_points * 0.7)
            random_count = max_points - systematic_count
            
            # Systematic sampling
            step = total_points // systematic_count
            systematic_indices = np.arange(0, total_points, step)[:systematic_count]
            
            # Random sampling from remaining points
            remaining_indices = np.setdiff1d(np.arange(total_points), systematic_indices)
            if len(remaining_indices) >= random_count:
                random_indices = np.random.choice(remaining_indices, random_count, replace=False)
                all_indices = np.concatenate([systematic_indices, random_indices])
            else:
                all_indices = systematic_indices
                
            return self.points[all_indices]
        else:  # < 50% - systematic sampling only
            step = int(1 / sample_ratio)
            indices = np.arange(0, total_points, step)[:max_points]
            return self.points[indices]
        
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
            'inlier_quantile': 0.80,
            'z_min': self.z_min_input.value(),
            'z_max': self.z_max_input.value()
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
            self.export_year_btn.setEnabled(True)  # NEW: Enable export year button
            self.display_results(self.current_results)
            
            # Auto-visualize all slices in Plots tab
            self.visualize_all_slices_in_plots()
            
            # Automatically visualize 3D slices if checkbox is checked
            if self.auto_visualize.isChecked():
                self.visualize_circular_slices()
                self.statusBar().showMessage(f"Analysis completed - {len(self.current_results)} slices visualized in both 2D and 3D")
            else:
                self.statusBar().showMessage(f"Analysis completed - {len(self.current_results)} slices visualized in 2D plots")
        else:
            self.statusBar().showMessage("Analysis failed")

    # NEW: Add method to compute residual profile
    def compute_residual_profile(self, result):
        """Compute theta and delta_r for a slice result, using boundary points."""
        if "boundary_points" not in result or not result["boundary_points"]:
            return None
        boundary_xy = np.array(result["boundary_points"])
        cx, cy, R = result["cx"], result["cy"], result["R"]
        
        dx = boundary_xy[:, 0] - cx
        dy = boundary_xy[:, 1] - cy
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
        r = np.hypot(dx, dy)
        delta_r = r - R
        
        # Sort by theta for smooth plotting
        order = np.argsort(theta)
        return theta[order], delta_r[order]

    def visualize_all_slices_in_plots(self):
        """Visualize residual profiles (Δr vs θ) for all analyzed slices in the Plots tab"""
        if not hasattr(self, 'current_results') or not self.current_results:
            return
        
        # Clear previous plots
        self.figure.clear()
        
        # Get slice thickness for title
        slice_thickness = self.slice_thickness.value()
        
        # Determine grid layout based on number of slices - REMOVED 16 plot limit
        num_slices = len(self.current_results)
        if num_slices <= 4:
            rows, cols = 2, 2
        elif num_slices <= 6:
            rows, cols = 2, 3
        elif num_slices <= 9:
            rows, cols = 3, 3
        elif num_slices <= 12:
            rows, cols = 3, 4
        elif num_slices <= 16:
            rows, cols = 4, 4
        elif num_slices <= 20:
            rows, cols = 4, 5
        elif num_slices <= 25:
            rows, cols = 5, 5
        elif num_slices <= 30:
            rows, cols = 5, 6
        elif num_slices <= 36:
            rows, cols = 6, 6
        else:
            # For very large numbers, use square root to get reasonable grid
            import math
            cols = math.ceil(math.sqrt(num_slices))
            rows = math.ceil(num_slices / cols)
            
        # Display ALL plots - REMOVED max_plots limitation
        max_plots = num_slices  # Show all slices
        
        # Increase figure size for more plots
        if num_slices > 16:
            self.figure.set_size_inches(12, 10)  # Larger figure for more plots
        
        self.statusBar().showMessage(f"Visualizing {max_plots} residual profiles...")
        
        # Ask user for PDF save location
        pdf_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Residual Profiles to PDF",
            f"residual_profiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf);;All Files (*.*)"
        )
        
        # Import PdfPages for multi-page PDF
        from matplotlib.backends.backend_pdf import PdfPages
        
        pdf_pages = None
        if pdf_path:
            try:
                pdf_pages = PdfPages(pdf_path)
            except Exception as e:
                self.statusBar().showMessage(f"Error creating PDF: {str(e)}")
                pdf_path = None
        
        # Create subplots for each slice
        for i, result in enumerate(self.current_results[:max_plots]):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            
            try:
                # Get slice data
                z_target = result['z_center']
                
                # Compute residual profile
                profile = self.compute_residual_profile(result)
                if profile is None:
                    ax.text(0.5, 0.5, 'No boundary data', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Z={z_target:.2f}m')
                    continue
                
                theta, delta_r = profile
                
                # Convert theta from radians to degrees
                theta_deg = np.degrees(theta)
                
                # Plot baseline (ideal circle) and residuals
                ax.axhline(0.0, color='k', linestyle='--', linewidth=1.5, label='Ideal circle (Δr=0)')
                ax.plot(theta_deg, delta_r, 'b-', linewidth=1, label='Δr = r - R')
                
                # Formatting
                ax.set_xlabel('θ (deg)', fontsize=6 if num_slices > 25 else 8)
                ax.set_ylabel('Δr', fontsize=6 if num_slices > 25 else 8)
                ax.set_title(f'Z={z_target:.2f}m, Thickness={slice_thickness:.3f}m', fontsize=6 if num_slices > 25 else 8)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6 if num_slices > 25 else 8)
                
                # Set ticks every 30 degrees
                ax.set_xticks(np.arange(0, 360, 30))
                
                # Adjust ticks for readability
                ax.tick_params(labelsize=5 if num_slices > 25 else 7)
                
                # Save individual plot to PDF if requested
                if pdf_pages:
                    # Create a separate figure for PDF page
                    fig_pdf = plt.figure(figsize=(8, 6))
                    ax_pdf = fig_pdf.add_subplot(1, 1, 1)
                    
                    # Replicate the plot for PDF
                    ax_pdf.axhline(0.0, color='k', linestyle='--', linewidth=1.5, label='Ideal circle (Δr=0)')
                    ax_pdf.plot(theta_deg, delta_r, 'b-', linewidth=1, label='Δr = r - R')
                    ax_pdf.set_xlabel('θ (deg)', fontsize=8)
                    ax_pdf.set_ylabel('Δr', fontsize=8)
                    ax_pdf.set_title(f'Z={z_target:.2f}m, Thickness={slice_thickness:.3f}m', fontsize=8)
                    ax_pdf.grid(True, alpha=0.3)
                    ax_pdf.legend(fontsize=8)
                    ax_pdf.set_xticks(np.arange(0, 360, 30))
                    
                    fig_pdf.tight_layout()
                    pdf_pages.savefig(fig_pdf)
                    plt.close(fig_pdf)
                
            except Exception as e:
                # If individual slice fails, show error
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=6)
                ax.set_title(f'Z={result["z_center"]:.2f}m - Error', fontsize=6)
        
        # Close PDF if created
        if pdf_pages:
            pdf_pages.close()
            self.statusBar().showMessage(f"PDF saved to {pdf_path}")
        
        # Adjust layout - Show analysis thickness in title
        title_fontsize = 10 if num_slices > 25 else 12
        self.figure.suptitle(f'Residual Profiles (Δr vs θ) - Analysis Thickness: {slice_thickness:.3f}m', 
                            fontsize=title_fontsize)
        
        # Tighter layout for many plots
        if num_slices > 16:
            self.figure.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
        else:
            self.figure.tight_layout()
        
        # Draw and switch to plots tab
        self.canvas.draw()
        self.results_tabs.setCurrentIndex(1)  # Switch to Plots tab
        
        self.statusBar().showMessage(f"All {max_plots} slices visualized in Plots tab (Thickness: {slice_thickness:.3f}m)")

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
            
        # Get values for all three axes
        x_values = self.points[:, 0]
        y_values = self.points[:, 1]
        z_values = self.points[:, 2]
        
        # Calculate statistics for each axis
        axes_data = {
            'X-axis': x_values,
            'Y-axis': y_values,
            'Z-axis': z_values
        }
        
        all_stats = {}
        for axis_name, values in axes_data.items():
            stats = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'range': np.max(values) - np.min(values)
            }
            all_stats[axis_name] = stats
        
        # Create statistics table with all three axes
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)  # Statistic name + 3 axes
        self.stats_table.setHorizontalHeaderLabels(["Statistic", "X-axis", "Y-axis", "Z-axis"])
        self.stats_table.setRowCount(7)  # 7 statistics
        
        # Statistics labels
        stat_labels = ['Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Range']
        stat_keys = ['count', 'mean', 'median', 'min', 'max', 'std', 'range']
        
        # Add statistics to table
        for i, (label, key) in enumerate(zip(stat_labels, stat_keys)):
            self.stats_table.setItem(i, 0, QTableWidgetItem(label))
            
            # Add values for each axis
            for j, axis_name in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
                value = all_stats[axis_name][key]
                if key == 'count':
                    self.stats_table.setItem(i, j+1, QTableWidgetItem(f"{int(value):,}"))
                else:
                    self.stats_table.setItem(i, j+1, QTableWidgetItem(f"{value:.3f}"))
        
        # Add to statistics tab (create if doesn't exist)
        if not hasattr(self, 'stats_tab'):
            self.stats_tab = QWidget()
            self.results_tabs.addTab(self.stats_tab, "Statistics")
            stats_layout = QVBoxLayout(self.stats_tab)
            
            # Add description label
            desc_label = QLabel("Point Cloud Statistics:")
            desc_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            stats_layout.addWidget(desc_label)
            
            # Add summary info
            self.summary_label = QLabel()
            self.summary_label.setStyleSheet("font-size: 11px; color: #666; margin: 10px 0;")
            stats_layout.addWidget(self.summary_label)
            
            stats_layout.addWidget(self.stats_table)

        else:
            # Update existing table
            stats_layout = self.stats_tab.layout()
            # Clear and recreate the table
            for i in reversed(range(stats_layout.count())):
                child = stats_layout.itemAt(i).widget()
                if isinstance(child, QTableWidget):
                    child.setParent(None)
            stats_layout.addWidget(self.stats_table)
        
        # Update summary info
        total_points = len(self.points)
        x_range = all_stats['X-axis']['range']
        y_range = all_stats['Y-axis']['range']
        z_range = all_stats['Z-axis']['range']
        
        # Determine which axis has the largest range (likely the main cylinder axis)
        ranges = [x_range, y_range, z_range]
        main_axis = ['X', 'Y', 'Z'][ranges.index(max(ranges))]
        
        summary_text = (
            f"Total Points: {total_points:,} | "
            f"X Range: {x_range:.3f}m | "
            f"Y Range: {y_range:.3f}m | "
            f"Z Range: {z_range:.3f}m | "
            f"Main Axis: {main_axis}"
        )
        
        if hasattr(self, 'summary_label'):
            self.summary_label.setText(summary_text)
        
        # Auto-resize columns
        self.stats_table.resizeColumnsToContents()
        
        # Show statistics in status bar
        self.statusBar().showMessage(
            f"Points: {total_points:,} | Main axis: {main_axis} | "
            f"{main_axis} range: {max(ranges):.3f}m"
        )

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
    
    # NEW: Add method to export data for a specific year
    def export_data_year(self):
        """Export data for a specific year with detailed boundary information for comparison"""
        if not hasattr(self, 'current_results'):
            self.statusBar().showMessage("No results to export")
            return
        
        # Get year from user input
        year, ok = QInputDialog.getText(
            self,
            "Enter Year",
            "Enter the year this data was scanned:",
            text=str(datetime.now().year)
        )
        
        if not ok or not year.strip():
            return
        
        try:
            year = year.strip()
            
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Year Data",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not save_dir:
                return
            
            # Prepare data for export
            export_data = []
            
            for result in self.current_results:
                z_center = result['z_center']
                cx = result['cx']
                cy = result['cy']
                R = result['R']
                thickness = self.slice_thickness.value()  # Get current slice thickness
                
                # Get boundary points and compute delta_r
                if 'boundary_points' in result and result['boundary_points']:
                    boundary_xy = np.array(result['boundary_points'])
                    
                    # Compute delta_r for each boundary point
                    dx = boundary_xy[:, 0] - cx
                    dy = boundary_xy[:, 1] - cy
                    r = np.hypot(dx, dy)
                    delta_r = r - R
                    
                    # Compute theta for ordering
                    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
                    
                    # Create rows for each boundary point
                    for i, (theta_val, delta_r_val) in enumerate(zip(theta, delta_r)):
                        export_data.append({
                            'year': year,
                            'z_center': z_center,
                            'cx': cx,
                            'cy': cy,
                            'R': R,
                            'thickness': thickness,
                            'theta': theta_val,
                            'delta_r': delta_r_val,
                            'boundary_point_index': i
                        })
                else:
                    # If no boundary points, still export slice info with NaN delta_r
                    export_data.append({
                        'year': year,
                        'z_center': z_center,
                        'cx': cx,
                        'cy': cy,
                        'R': R,
                        'thickness': thickness,
                        'theta': np.nan,
                        'delta_r': np.nan,
                        'boundary_point_index': 0
                    })
            
            # Create DataFrame and save
            df_export = pd.DataFrame(export_data)
            export_path = os.path.join(save_dir, f"year_{year}_boundary_data.csv")
            df_export.to_csv(export_path, index=False, float_format='%.6f')
            
            # Also save summary statistics for the year
            summary_data = []
            for result in self.current_results:
                summary_data.append({
                    'year': year,
                    'z_center': result['z_center'],
                    'cx': result['cx'],
                    'cy': result['cy'],
                    'R': result['R'],
                    'thickness': self.slice_thickness.value(),
                    'ovality_pct': result.get('ovality_pct', np.nan),
                    'n_boundary_points': len(result.get('boundary_points', []))
                })
            
            df_summary = pd.DataFrame(summary_data)
            summary_path = os.path.join(save_dir, f"year_{year}_summary.csv")
            df_summary.to_csv(summary_path, index=False, float_format='%.6f')
            
            self.statusBar().showMessage(
                f"Year {year} data exported to {save_dir} - {len(export_data)} boundary points"
            )
            
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting year data: {str(e)}")

    def visualize_slice(self):
        if self.points is None:
            self.statusBar().showMessage("Please load point cloud first")
            return
            
        z_target = self.z_slice_input.value()
        slice_thickness = self.viz_slice_thickness.value()  # Use visualization thickness
        dz = slice_thickness / 2.0
        
        # Get slice points
        mask = (self.points[:, 2] >= z_target - dz) & (self.points[:, 2] <= z_target + dz)
        slice_points = self.points[mask]
        
        # Show point count info
        self.statusBar().showMessage(f"Found {len(slice_points)} points in slice (thickness: {slice_thickness:.3f}m)")
        
        if len(slice_points) < 100:  # Lower threshold since we're just visualizing
            self.statusBar().showMessage(f"Too few points ({len(slice_points)}) in slice - try increasing thickness")
            return
            
        # Configure parameters with visualization thickness
        params = {
            'window_len': slice_thickness,  # Use visualization thickness
            'z_step': self.z_step.value(),
            'boundary_method': self.boundary_method.currentText(),
            'angle_bins': self.angle_bins.value(),
            'max_points_for_speed': 1_000_000,
            'min_points_per_slice': 100,  # Lower threshold for visualization
            'inlier_quantile': 0.80
        }
        
        try:
            # Process single slice
            analyzer = CylinderAnalyzer(Config(**params))
            result = analyzer.process_slice(slice_points, z_target)
            
            # Check if result is None first
            if result is None:
                self.statusBar().showMessage("Could not process slice - try adjusting parameters or increasing thickness")
                return
            
            # Debug: print detailed result structure
            print(f"Debug - Result type: {type(result)}, Length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            # Initialize default values
            xc = yc = R = 0.0
            edge_xy = None
            ovality_pct = 0.0
            result_dict = None
            
            # Handle the actual result format based on debug output
            try:
                if isinstance(result, tuple) and len(result) == 3:
                    # Format: (inner_tuple, dataframe, None)
                    inner_tuple, df, _ = result
                    
                    print(f"Debug - Inner tuple length: {len(inner_tuple)}")
                    
                    if isinstance(inner_tuple, tuple) and len(inner_tuple) >= 4:
                        # Expected format: (result_dict, edge_array, boundary_array, (cx, cy, R), ovality_dict)
                        if len(inner_tuple) == 5:
                            result_dict, edge_xy, _, circle_params, ovality_dict = inner_tuple
                        elif len(inner_tuple) == 4:
                            result_dict, edge_xy, _, circle_params = inner_tuple
                            ovality_dict = {}
                        else:
                            # Try to get the last 2 elements as circle_params and ovality
                            result_dict = inner_tuple[0]
                            edge_xy = inner_tuple[1] if isinstance(inner_tuple[1], np.ndarray) else None
                            circle_params = inner_tuple[-2] if len(inner_tuple) >= 2 else None
                            ovality_dict = inner_tuple[-1] if len(inner_tuple) >= 1 else {}
                        
                        # Extract circle parameters
                        if isinstance(circle_params, tuple) and len(circle_params) == 3:
                            xc, yc, R = circle_params
                            print(f"Debug - Circle params: cx={xc}, cy={yc}, R={R}")
                        else:
                            # Try to get from result_dict
                            if isinstance(result_dict, dict):
                                xc = result_dict.get('cx', 0.0)
                                yc = result_dict.get('cy', 0.0) 
                                R = result_dict.get('R', 0.0)
                                print(f"Debug - Circle params from dict: cx={xc}, cy={yc}, R={R}")
                            else:
                                raise ValueError("Cannot extract circle parameters")
                        
                        # Extract ovality
                        if isinstance(ovality_dict, dict):
                            ovality_pct = ovality_dict.get('ovality_pct', 0.0)
                        elif isinstance(result_dict, dict):
                            ovality_pct = result_dict.get('ovality_pct', 0.0)
                        else:
                            ovality_pct = 0.0
                            
                    else:
                        raise ValueError(f"Inner tuple has unexpected length: {len(inner_tuple)}")
                        
                else:
                    raise ValueError(f"Unexpected result format with {len(result) if hasattr(result, '__len__') else 0} elements")
                    
            except Exception as parse_error:
                print(f"Debug - Error parsing result: {parse_error}")
                self.statusBar().showMessage(f"Error parsing analysis result: {str(parse_error)}")
                return

            # Validate extracted values
            if R <= 0:
                self.statusBar().showMessage("Invalid radius calculated - try increasing parameters")
                return

            print(f"Debug - Final values: xc={xc}, yc={yc}, R={R}, ovality={ovality_pct}")

            # Extract inlier_mask if available
            inlier_mask = None
            if isinstance(result_dict, dict) and "inlier_mask" in result_dict:
                inlier_mask = np.array(result_dict["inlier_mask"], dtype=bool)
                print(f"Debug - Using inlier mask with {np.sum(inlier_mask)}/{len(inlier_mask)} inliers")

            # Clear previous plot
            self.slice_figure.clear()
            
            # Create 1x2 subplots
            ax1 = self.slice_figure.add_subplot(1, 2, 1)  # Left: XY view
            ax2 = self.slice_figure.add_subplot(1, 2, 2)  # Right: Residual profile
            
            # Plot points (subsample if too many)
            if len(slice_points) > 30000:
                idx = np.random.choice(len(slice_points), 30000, replace=False)
                plot_points = slice_points[idx]
            else:
                plot_points = slice_points
            
            # Left subplot: XY view
            ax1.scatter(plot_points[:, 0], plot_points[:, 1], 
                    s=1, alpha=0.3, label='Points', c='lightblue')
            
            if edge_xy is not None and len(edge_xy) > 0:
                ax1.plot(edge_xy[:, 0], edge_xy[:, 1], 'k-', 
                    linewidth=1.5, label='Boundary')
            
            # Plot fitted circle
            theta = np.linspace(0, 2*np.pi, 360)
            circle_x = xc + R*np.cos(theta)
            circle_y = yc + R*np.sin(theta)
            ax1.plot(circle_x, circle_y, 'r-', 
                linewidth=2, label='Fitted Circle')
            
            # Plot center
            ax1.plot(xc, yc, 'r+', markersize=12, markeredgewidth=3, label='Center')
            
            ax1.set_aspect('equal')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('XY View')
            ax1.legend()
            
            # Right subplot: Residual profile (Δr vs θ)
            if edge_xy is not None and len(edge_xy) > 0:
                # Compute residual profile
                dx = edge_xy[:, 0] - xc
                dy = edge_xy[:, 1] - yc
                theta_res = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
                r = np.hypot(dx, dy)
                delta_r = r - R
                
                # MODIFIED: Filter outliers using inlier_mask or dynamic quantile filtering
                if inlier_mask is not None and len(inlier_mask) == len(edge_xy):
                    # Use inlier mask from hybrid fit (preferred)
                    valid_mask = inlier_mask
                    print(f"Debug - Filtered to {np.sum(valid_mask)} inlier points")
                else:
                    # Fallback: Dynamic outlier filtering using quantile (remove top 5% outliers)
                    abs_delta_r = np.abs(delta_r)
                    threshold = np.quantile(abs_delta_r, 0.95)  # 95th percentile as threshold
                    valid_mask = abs_delta_r <= threshold
                    print(f"Debug - Fallback filtering: removed {np.sum(~valid_mask)} outliers using quantile threshold {threshold:.3f}")
                
                # Apply filtering
                theta_filtered = theta_res[valid_mask]
                delta_r_filtered = delta_r[valid_mask]
                
                # Sort by theta for smooth plotting
                order = np.argsort(theta_filtered)
                theta_sorted = theta_filtered[order]
                delta_r_sorted = delta_r_filtered[order]
                
                # Convert theta from radians to degrees
                theta_sorted_deg = np.degrees(theta_sorted)
                
                # Plot baseline and residuals
                ax2.axhline(0.0, color='k', linestyle='--', linewidth=1.5, label='Ideal circle (Δr=0)')
                ax2.plot(theta_sorted_deg, delta_r_sorted, 'b-', linewidth=1, label='Δr = r - R')
                
                # Add red dots at each boundary point position
                ax2.scatter(theta_sorted_deg, delta_r_sorted, color='red', s=8, alpha=0.8, label='Boundary points')
                
                # Add info about filtering
                ax2.text(0.02, 0.98, f'Filtered: {len(theta_sorted)}/{len(edge_xy)} points', 
                        transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax2.set_xlabel('θ (deg)')
                ax2.set_ylabel('Δr')
                ax2.set_title('Residual Profile (Outliers Filtered)')
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(np.arange(0, 360, 30))
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'No boundary data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Residual Profile')
            
            # Overall title and layout
            self.slice_figure.suptitle(f'Slice at Z = {z_target:.4f}m (±{dz:.4f}m) - Thickness: {slice_thickness:.3f}m')
            self.slice_figure.tight_layout()
            
            self.slice_canvas.draw()
            
            # Switch to slice tab
            self.results_tabs.setCurrentIndex(2)
            
            self.statusBar().showMessage(f"Slice visualization complete - {len(slice_points):,} points, thickness: {slice_thickness:.3f}m")
            
            # After successful processing, store the slice result
            self.current_slice_result = {
                'z_target': z_target,
                'slice_thickness': slice_thickness,
                'xc': xc,
                'yc': yc,
                'R': R,
                'ovality_pct': ovality_pct,
                'edge_xy': edge_xy,
                'theta_sorted': theta_sorted if 'theta_sorted' in locals() else None,
                'delta_r_sorted': delta_r_sorted if 'delta_r_sorted' in locals() else None,
                'n_points': len(slice_points)
            }
            
            # Enable export slice button
            self.export_slice_btn.setEnabled(True)
            
        except Exception as e:
            error_msg = str(e)
            print(f"Debug - Full error in visualize_slice: {error_msg}")
            print(f"Debug - Error type: {type(e)}")
            
            # More specific error messages
            if "cannot unpack" in error_msg:
                self.statusBar().showMessage("Analysis failed - try increasing slice thickness or adjusting parameters")
            elif "process_slice" in error_msg:
                self.statusBar().showMessage("Slice processing failed - check data quality and parameters")
            else:
                self.statusBar().showMessage(f"Error: {error_msg} - try increasing thickness")

    def visualize_circular_slices(self):
        """Visualize circular slices in 3D VTK view based on analysis results"""
        if not hasattr(self, 'current_results') or not self.current_results:
            return
        
        # Clear existing slice actors
        if hasattr(self, 'slice_actors'):
            for actor in self.slice_actors:
                self.renderer_slices.RemoveActor(actor)
        
        self.slice_actors = []
        
        # Get slice thickness
        thickness = self.slice_thickness.value()
        
        # Determine data orientation
        if hasattr(self, 'points') and self.points is not None:
            # Calculate ranges to determine main axis
            x_range = np.max(self.points[:, 0]) - np.min(self.points[:, 0])
            y_range = np.max(self.points[:, 1]) - np.min(self.points[:, 1]) 
            z_range = np.max(self.points[:, 2]) - np.min(self.points[:, 2])
            
            print(f"Data ranges - X: {x_range:.3f}, Y: {y_range:.3f}, Z: {z_range:.3f}")
            
            # Find main axis (longest dimension)
            ranges = [x_range, y_range, z_range]
            main_axis = ranges.index(max(ranges))  # 0=X, 1=Y, 2=Z
            print(f"Main axis detected: {['X','Y','Z'][main_axis]}")
        else:
            main_axis = 1  # Default to Y
        
        # Create circular slices for each result
        for result in self.current_results:
            cx = result['cx']
            cy = result['cy']
            radius = result['R']
            z_center = result['z_center']
            
            # Create disk for flat slice
            disk = vtk.vtkDiskSource()
            disk.SetInnerRadius(0.0)
            disk.SetOuterRadius(radius)
            disk.SetRadialResolution(36)
            disk.SetCircumferentialResolution(60)
            
            # Extrude to create thickness
            extrude = vtk.vtkLinearExtrusionFilter()
            extrude.SetInputConnection(disk.GetOutputPort())
            extrude.SetExtrusionTypeToNormalExtrusion()
            extrude.SetScaleFactor(thickness)
            
            # Transform based on detected main axis
            transform = vtk.vtkTransform()
            
            if main_axis == 1:  # Y is main axis - cylinder along Y
                # Disk is in XY plane by default, need to rotate to XZ plane (perpendicular to Y)
                transform.RotateX(90)  # Rotate disk to be perpendicular to Y-axis
                transform.Translate(cx, z_center, cy)
            elif main_axis == 2:  # Z is main axis - cylinder along Z  
                # Disk already in XY plane (perpendicular to Z), just translate
                transform.Translate(cx, cy, z_center)
            else:  # X is main axis - cylinder along X
                transform.RotateY(90)  # Rotate disk to be perpendicular to X-axis
                transform.Translate(z_center, cx, cy)
            
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetInputConnection(extrude.GetOutputPort())
            transform_filter.SetTransform(transform)
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(transform_filter.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set color based on position along main axis
            if main_axis == 1:
                pos_values = [r['z_center'] for r in self.current_results]
            elif main_axis == 2:
                pos_values = [r['z_center'] for r in self.current_results]
            else:
                pos_values = [r['z_center'] for r in self.current_results]
                
            pos_min = min(pos_values)
            pos_max = max(pos_values)
            
            if pos_max != pos_min:
                t = (z_center - pos_min) / (pos_max - pos_min)
            else:
                t = 0.5
                
            # Color gradient: blue -> green -> red
            r = min(1.0, 2.0 * t)
            g = min(1.0, 2.0 * (1.0 - abs(t - 0.5)))
            b = min(1.0, 2.0 * (1.0 - t))
            
            actor.GetProperty().SetColor(r, g, b)
            actor.GetProperty().SetOpacity(0.6)
            
            # Add to slice renderer
            self.renderer_slices.AddActor(actor)
            self.slice_actors.append(actor)
        
        # Update camera and render
        self.renderer_slices.ResetCamera()
        self.vtk_widget_slices.GetRenderWindow().Render()
        
        self.statusBar().showMessage(f"Visualized {len(self.current_results)} slices along {['X','Y','Z'][main_axis]} axis")

    def compare_years(self):
        """Compare current slice with data from multiple years' files"""
        # Check if we have current slice data
        if not hasattr(self, 'current_slice_result') or self.current_slice_result is None:
            self.statusBar().showMessage("Please visualize a slice first before comparing")
            return
        
        # Ask for number of years to compare
        num_years, ok = QInputDialog.getInt(
            self,
            "Number of Years",
            "Enter number of years to compare (minimum 1):",
            value=1,
            min=1,
            max=10
        )
        
        if not ok:
            return
        
        # Create dialog for file selection
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select Slice Files for {num_years} Years")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Select boundary_data.csv files for {num_years} years:"))
        
        # Create file selectors
        file_buttons = []
        file_labels = []
        
        for i in range(num_years):
            hbox = QHBoxLayout()
            label = QLabel(f"Year {i+1}: No file selected")
            file_labels.append(label)
            hbox.addWidget(label)
            
            btn = QPushButton("Browse...")
            btn.clicked.connect(lambda checked, idx=i: self.select_comparison_file(idx, file_labels, file_buttons))
            file_buttons.append(btn)
            hbox.addWidget(btn)
            
            layout.addLayout(hbox)
        
        # OK/Cancel buttons
        button_box = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_box.addWidget(ok_btn)
        button_box.addWidget(cancel_btn)
        layout.addLayout(button_box)
        
        # Store file paths
        self.comparison_files = [None] * num_years
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Check if all files selected
            if None in self.comparison_files:
                self.statusBar().showMessage("Please select files for all years")
                return
            
            try:
                # Get current slice parameters
                current_z = self.current_slice_result['z_target']
                current_thickness = self.current_slice_result['slice_thickness']
                
                # Clear slice figure and create comparison plot
                self.slice_figure.clear()
                
                # Create subplot for residual profiles
                ax = self.slice_figure.add_subplot(1, 1, 1)
                
                # Define colors for different years (including current)
                colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
                
                # Plot current slice data first (blue)
                current_result = self.current_slice_result
                if current_result['theta_sorted'] is not None and current_result['delta_r_sorted'] is not None:
                    ax.plot(current_result['theta_sorted'], current_result['delta_r_sorted'], 
                           color='blue', linewidth=2, label='Current Slice', alpha=0.8)
                    ax.scatter(current_result['theta_sorted'], current_result['delta_r_sorted'], 
                              color='blue', s=10, alpha=0.6)
                
                # Load and plot each comparison file
                valid_comparisons = 0
                for i, file_path in enumerate(self.comparison_files):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Validate file format
                        required_cols = ['year', 'z_center', 'thickness', 'theta', 'delta_r']
                        if not all(col in df.columns for col in required_cols):
                            self.statusBar().showMessage(f"Skipping {os.path.basename(file_path)}: invalid format")
                            continue
                        
                        # Filter data for matching z_center and thickness
                        matching_slices = df[
                            (np.isclose(df['z_center'], current_z, atol=1e-3)) & 
                            (np.isclose(df['thickness'], current_thickness, atol=1e-3))
                        ]
                        
                        if matching_slices.empty:
                            self.statusBar().showMessage(f"Skipping {os.path.basename(file_path)}: no matching slice")
                            continue
                        
                        # Get year from matching data
                        comparison_year = str(matching_slices['year'].iloc[0])
                        
                        # Plot comparison data
                        comparison_df_sorted = matching_slices.sort_values('theta')
                        ax.plot(comparison_df_sorted['theta'], comparison_df_sorted['delta_r'], 
                               color=colors[i+1], linewidth=2, label=f'Year {comparison_year}', alpha=0.8)
                        ax.scatter(comparison_df_sorted['theta'], comparison_df_sorted['delta_r'], 
                                  color=colors[i+1], s=10, alpha=0.6)
                        
                        valid_comparisons += 1
                        
                    except Exception as e:
                        self.statusBar().showMessage(f"Error loading {os.path.basename(file_path)}: {str(e)}")
                        continue
                
                if valid_comparisons == 0:
                    self.statusBar().showMessage("No valid comparison data found")
                    return
                
                # Add baseline
                ax.axhline(0.0, color='k', linestyle='--', linewidth=1.5, label='Ideal circle (Δr=0)')
                
                ax.set_xlabel('θ (rad)')
                ax.set_ylabel('Δr')
                ax.set_title(f'Residual Profile Comparison - Current vs {valid_comparisons} Years\n'
                            f'Z: {current_z:.3f}m, Thickness: {current_thickness:.3f}m')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                self.slice_figure.tight_layout()
                self.slice_canvas.draw()
                
                # Switch to slice tab
                self.results_tabs.setCurrentIndex(2)
                
                self.statusBar().showMessage(f"Comparison completed - Current slice vs {valid_comparisons} years")
                
            except Exception as e:
                self.statusBar().showMessage(f"Error during comparison: {str(e)}")
    
    def select_comparison_file(self, index, labels, buttons):
        """Select file for comparison"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Boundary Data File for Year {index + 1}",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if filename:
            self.comparison_files[index] = filename
            labels[index].setText(f"Year {index + 1}: {os.path.basename(filename)}")

    # NEW: Add method to export current slice data
    def export_slice(self):
        """Export data for the current visualized slice"""
        if not hasattr(self, 'current_slice_result'):
            self.statusBar().showMessage("No slice data to export")
            return
        
        # Get year from user input
        year, ok = QInputDialog.getText(
            self,
            "Enter Year",
            "Enter the year this data was scanned:",
            text=str(datetime.now().year)
        )
        
        if not ok or not year.strip():
            return
        
        try:
            year = year.strip()
            
            # Get directory to save files
            save_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Directory to Save Slice Data",
                "",
                QFileDialog.Option.ShowDirsOnly
            )
            
            if not save_dir:
                return
            
            result = self.current_slice_result
            
            # Prepare boundary data for export
            export_data = []
            
            if result['edge_xy'] is not None and len(result['edge_xy']) > 0 and result['theta_sorted'] is not None and result['delta_r_sorted'] is not None:
                # Create rows for each boundary point
                for i, (theta_val, delta_r_val) in enumerate(zip(result['theta_sorted'], result['delta_r_sorted'])):
                    export_data.append({
                        'year': year,
                        'z_center': result['z_target'],
                        'cx': result['xc'],
                        'cy': result['yc'],
                        'R': result['R'],
                        'thickness': result['slice_thickness'],
                        'theta': theta_val,
                        'delta_r': delta_r_val,
                        'boundary_point_index': i
                    })
            else:
                # If no boundary points, still export slice info with NaN delta_r
                export_data.append({
                    'year': year,
                    'z_center': result['z_target'],
                    'cx': result['xc'],
                    'cy': result['yc'],
                    'R': result['R'],
                    'thickness': result['slice_thickness'],
                    'theta': np.nan,
                    'delta_r': np.nan,
                    'boundary_point_index': 0
                })
            
            # Create DataFrame and save
            df_export = pd.DataFrame(export_data)
            export_path = os.path.join(save_dir, f"slice_{year}_{result['z_target']:.3f}m_{result['slice_thickness']:.3f}m_boundary_data.csv")
            df_export.to_csv(export_path, index=False, float_format='%.6f')
            
            self.statusBar().showMessage(
                f"Slice data exported to {save_dir} - {len(export_data)} boundary points"
            )
            
        except Exception as e:
            self.statusBar().showMessage(f"Error exporting slice data: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CylinderAnalyzerGUI()
    window.show()
    sys.exit(app.exec())