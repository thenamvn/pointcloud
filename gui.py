import sys
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
                            QTableWidget, QTableWidgetItem, QTabWidget, QSplitter, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from analyzer import CylinderAnalyzer, Config
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
            
            # Parse with progress updates
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
        self.renderer_slices.SetBackground(0.1, 0.1, 0.2)
        
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
                points_mb = self.points.nbytes / (1024 * 1024)
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
        
        if not filenames:
            return
        
        # Check if we already have data and determine mode
        if hasattr(self, 'points') and self.points is not None:
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Question)
            msg.setWindowTitle("Data Already Loaded")
            msg.setText(f"You already have {len(self.points):,} points loaded.")
            msg.setInformativeText(f"Selected {len(filenames)} new file(s). What would you like to do?")
            
            append_button = msg.addButton("Append all files to existing data", QMessageBox.ButtonRole.YesRole)
            replace_button = msg.addButton("Replace with new files", QMessageBox.ButtonRole.NoRole)
            cancel_button = msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            
            msg.setDefaultButton(append_button)
            result = msg.exec()
            clicked_button = msg.clickedButton()
            
            if clicked_button == append_button:
                self.append_mode = True
            elif clicked_button == replace_button:
                self.append_mode = False
                # Clear existing data first
                if hasattr(self, 'points'):
                    del self.points
                    self.points = None
                    import gc
                    gc.collect()
            else:
                return
        else:
            self.append_mode = False
        
        # Start incremental loading
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

    def visualize_all_slices_in_plots(self):
        """Visualize all analyzed slices in the Plots tab"""
        if not hasattr(self, 'current_results') or not self.current_results:
            return
        
        # Clear previous plots
        self.figure.clear()
        
        # Get thickness for visualization - Use analysis slice thickness, not viz thickness
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
        
        self.statusBar().showMessage(f"Visualizing {max_plots} slices...")
        
        # Create subplots for each slice
        for i, result in enumerate(self.current_results[:max_plots]):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            
            try:
                # Get slice data
                z_target = result['z_center']
                cx = result['cx']
                cy = result['cy']
                radius = result['R']
                ovality_pct = result.get('ovality_pct', 0.0)
                
                dz = slice_thickness / 2.0
                
                # Get slice points
                mask = (self.points[:, 2] >= z_target - dz) & (self.points[:, 2] <= z_target + dz)
                slice_points = self.points[mask]
                
                if len(slice_points) < 50:  # Skip if too few points
                    ax.text(0.5, 0.5, f'Too few points\n({len(slice_points)})', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Z={z_target:.2f}m')
                    continue
                
                # Subsample points for faster plotting - reduce for many plots
                max_points = 3000 if num_slices > 20 else 5000
                if len(slice_points) > max_points:
                    idx = np.random.choice(len(slice_points), max_points, replace=False)
                    plot_points = slice_points[idx]
                else:
                    plot_points = slice_points
                
                # Plot points - smaller points for many plots
                point_size = 0.3 if num_slices > 20 else 0.5
                ax.scatter(plot_points[:, 0], plot_points[:, 1], 
                        s=point_size, alpha=0.4, c='lightblue', rasterized=True)
                
                # Try to get and plot boundary if available
                try:
                    # Re-analyze this slice to get boundary - Use analysis parameters
                    params = {
                        'window_len': slice_thickness,
                        'z_step': self.z_step.value(),
                        'boundary_method': self.boundary_method.currentText(),
                        'angle_bins': self.angle_bins.value(),
                        'max_points_for_speed': 1_000_000,
                        'min_points_per_slice': 50,
                        'inlier_quantile': 0.80
                    }
                    
                    analyzer = CylinderAnalyzer(Config(**params))
                    slice_result = analyzer.process_slice(slice_points, z_target)
                    
                    if slice_result and len(slice_result) >= 3:
                        inner_tuple = slice_result[0]
                        if isinstance(inner_tuple, tuple) and len(inner_tuple) >= 2:
                            edge_xy = inner_tuple[1]
                            if edge_xy is not None and len(edge_xy) > 0:
                                line_width = 0.8 if num_slices > 20 else 1.0
                                ax.plot(edge_xy[:, 0], edge_xy[:, 1], 'k-', 
                                    linewidth=line_width, alpha=0.8)
                except:
                    pass  # Skip boundary if failed
                
                # Plot fitted circle
                theta = np.linspace(0, 2*np.pi, 360)
                circle_x = cx + radius * np.cos(theta)
                circle_y = cy + radius * np.sin(theta)
                circle_width = 1.0 if num_slices > 20 else 1.5
                ax.plot(circle_x, circle_y, 'r-', linewidth=circle_width, alpha=0.9)
                
                # Plot center
                marker_size = 6 if num_slices > 20 else 8
                ax.plot(cx, cy, 'r+', markersize=marker_size, markeredgewidth=2)
                
                # Set equal aspect and limits
                ax.set_aspect('equal')
                
                # Set title with key information - smaller font for many plots
                title_fontsize = 6 if num_slices > 25 else 8
                ax.set_title(f'Z={z_target:.2f}m\nR={radius:.3f}m, O={ovality_pct:.1f}%', 
                            fontsize=title_fontsize)
                
                # Minimal axis labels for space - only for bottom and left edges
                label_fontsize = 6 if num_slices > 25 else 8
                if i >= num_slices - cols:  # Bottom row
                    ax.set_xlabel('X (m)', fontsize=label_fontsize)
                if i % cols == 0:  # Left column
                    ax.set_ylabel('Y (m)', fontsize=label_fontsize)
                    
                # Smaller tick labels
                tick_fontsize = 5 if num_slices > 25 else 7
                ax.tick_params(labelsize=tick_fontsize)
                
                # Add grid
                ax.grid(True, alpha=0.2 if num_slices > 20 else 0.3)
                
            except Exception as e:
                # If individual slice fails, show error
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:20]}...', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=6)
                ax.set_title(f'Z={result["z_center"]:.2f}m - Error', fontsize=6)
        
        # Adjust layout - Show analysis thickness in title
        title_fontsize = 10 if num_slices > 25 else 12
        self.figure.suptitle(f'All Analyzed Slices (Analysis Thickness: {slice_thickness:.3f}m)', 
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
    
    # Add this method to CylinderAnalyzerGUI class

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
                self.statusBar().showMessage("Invalid radius calculated - try adjusting parameters")
                return
        
            print(f"Debug - Final values: xc={xc}, yc={yc}, R={R}, ovality={ovality_pct}")
        
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
                    s=1, alpha=0.3, label='Points', c='lightblue')
            
            if edge_xy is not None and len(edge_xy) > 0:
                ax.plot(edge_xy[:, 0], edge_xy[:, 1], 'k-', 
                    linewidth=1.5, label='Boundary')
            
            # Plot fitted circle
            theta = np.linspace(0, 2*np.pi, 360)
            circle_x = xc + R*np.cos(theta)
            circle_y = yc + R*np.sin(theta)
            ax.plot(circle_x, circle_y, 'r-', 
                linewidth=2, label='Fitted Circle')
            
            # Plot center
            ax.plot(xc, yc, 'r+', markersize=12, markeredgewidth=3, label='Center')
            
            # Add text with parameters (in meters)
            info_text = (
                f"Z = {z_target:.4f} ± {dz:.4f} m\n"
                f"Points: {len(slice_points):,}\n"
                f"Edge Points: {len(edge_xy) if edge_xy is not None else 0}\n"
                f"Center: ({xc:.4f}, {yc:.4f}) m\n"
                f"Radius: {R:.4f} m\n"
                f"Ovality: {ovality_pct:.3f}%\n"
                f"Viz Thickness: {slice_thickness:.3f} m"
            )
            ax.text(0.02, 0.98, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Slice Visualization at Z = {z_target:.4f}m (±{dz:.4f}m)')
            
            # Adjust layout before adding legend
            self.slice_figure.tight_layout()
            
            # Add legend with better placement
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust subplot parameters to fit legend
            self.slice_figure.subplots_adjust(right=0.85)
            
            self.slice_canvas.draw()
            
            # Switch to slice tab
            self.results_tabs.setCurrentIndex(2)
            
            self.statusBar().showMessage(f"Slice visualization complete - {len(slice_points):,} points, thickness: {slice_thickness:.3f}m")
            
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
            # Create separate tab for comparison if not exists
            if not hasattr(self, 'comparison_tab'):
                self.comparison_tab = QWidget()
                self.results_tabs.addTab(self.comparison_tab, "Year Comparison")
                comparison_layout = QVBoxLayout(self.comparison_tab)
                
                # Create separate figure for comparison
                self.comparison_figure = plt.figure(figsize=(10, 8))
                self.comparison_canvas = FigureCanvasQTAgg(self.comparison_figure)
                comparison_layout.addWidget(self.comparison_canvas)
            
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
            
            # Clear comparison plots
            self.comparison_figure.clear()
            
            # Plot 1: Ovality vs Z for different years
            ax1 = self.comparison_figure.add_subplot(211)
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
            ax2 = self.comparison_figure.add_subplot(212)
            for year in combined_df['Year'].unique():
                year_data = combined_df[combined_df['Year'] == year]
                ax2.plot(year_data['z_center'], year_data['R'], 
                        '-o', label=f'Year {year}', markersize=4)
            
            ax2.set_xlabel('Z Position')
            ax2.set_ylabel('Radius')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Radius Comparison')
            
            self.comparison_figure.tight_layout()
            self.comparison_canvas.draw()
            
            # Switch to comparison tab
            comparison_tab_index = self.results_tabs.indexOf(self.comparison_tab)
            self.results_tabs.setCurrentIndex(comparison_tab_index)
            
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