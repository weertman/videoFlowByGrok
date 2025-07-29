from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QComboBox, QCheckBox,
    QProgressBar, QSplitter, QFileDialog, QGroupBox, QMessageBox,
    QStatusBar, QInputDialog
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtCore import Qt, QTimer, Signal
import cv2
import numpy as np
import os
from src.worker import VideoLoaderThread, FlowProcessorThread
from src.utils import get_video_metadata, visualize_flow

class MainUI(QMainWindow):
    frame_ready = Signal(np.ndarray)  # For preview updates

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optical Flow Visualizer")
        self.setGeometry(100, 100, 1200, 800)
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.metadata = {}
        self.current_frame = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.thread = None
        self.processed_viz = []

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel: Input and Selection
        left_panel = QVBoxLayout()
        self.input_group = QGroupBox("Video Input")
        input_layout = QVBoxLayout(self.input_group)
        self.browse_button = QPushButton("Browse Video")
        self.browse_button.clicked.connect(self.load_video)
        input_layout.addWidget(self.browse_button)
        self.metadata_label = QLabel("No video loaded")
        input_layout.addWidget(self.metadata_label)
        self.preview_label = QLabel("Video Preview")
        self.preview_label.setFixedSize(320, 240)
        self.preview_label.setAcceptDrops(True)  # Enable drag-drop
        input_layout.addWidget(self.preview_label)
        playback_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        playback_layout.addWidget(self.play_button)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.sliderMoved.connect(self.seek_video)
        playback_layout.addWidget(self.seek_slider)
        input_layout.addLayout(playback_layout)
        left_panel.addWidget(self.input_group)

        self.frame_group = QGroupBox("Frame Selection")
        frame_layout = QGridLayout(self.frame_group)
        frame_layout.addWidget(QLabel("Start Frame:"), 0, 0)
        self.start_spin = QSpinBox()
        frame_layout.addWidget(self.start_spin, 0, 1)
        frame_layout.addWidget(QLabel("End Frame:"), 1, 0)
        self.end_spin = QSpinBox()
        frame_layout.addWidget(self.end_spin, 1, 1)
        self.entire_check = QCheckBox("Process Entire Video")
        self.entire_check.stateChanged.connect(self.toggle_entire)
        frame_layout.addWidget(self.entire_check, 2, 0, 1, 2)
        frame_layout.addWidget(QLabel("Frame Step:"), 3, 0)
        self.step_spin = QSpinBox(value=1, minimum=1)
        frame_layout.addWidget(self.step_spin, 3, 1)
        left_panel.addWidget(self.frame_group)
        main_layout.addLayout(left_panel)

        # Right panel: Parameters and Output
        right_splitter = QSplitter(Qt.Vertical)
        self.param_group = QGroupBox("Flow Visualization Parameters")
        param_layout = QGridLayout(self.param_group)
        self.algo_label = QLabel("Algorithm:")
        param_layout.addWidget(self.algo_label, 0, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["Farneback", "Dense Optical Flow", "FlowTrace"])
        param_layout.addWidget(self.algo_combo, 0, 1)
        self.algo_combo.currentTextChanged.connect(self.update_params_visibility)
        # Placeholder for dynamic params (use QStackedWidget in future)
        self.win_size_label = QLabel("Window Size:")
        param_layout.addWidget(self.win_size_label, 1, 0)
        self.win_size_spin = QSpinBox(value=15, minimum=1)
        param_layout.addWidget(self.win_size_spin, 1, 1)
        self.pyr_levels_label = QLabel("Pyramid Levels:")
        param_layout.addWidget(self.pyr_levels_label, 2, 0)
        self.pyr_levels_spin = QSpinBox(value=3, minimum=1)
        param_layout.addWidget(self.pyr_levels_spin, 2, 1)
        self.iter_label = QLabel("Iterations:")
        param_layout.addWidget(self.iter_label, 3, 0)
        self.iter_spin = QSpinBox(value=3, minimum=1)
        param_layout.addWidget(self.iter_spin, 3, 1)
        self.mag_label = QLabel("Magnitude Threshold:")
        param_layout.addWidget(self.mag_label, 4, 0)
        self.mag_slider = QSlider(Qt.Horizontal)
        self.mag_slider.setRange(0, 100)
        self.mag_slider.setValue(0)
        param_layout.addWidget(self.mag_slider, 4, 1)
        self.cmap_label = QLabel("Color Map:")
        param_layout.addWidget(self.cmap_label, 5, 0)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["hsv", "jet", "viridis", "grayscale"])
        param_layout.addWidget(self.cmap_combo, 5, 1)
        self.arrow_density_label = QLabel("Arrow Density:")
        param_layout.addWidget(self.arrow_density_label, 6, 0)
        self.arrow_density_spin = QSpinBox(value=16, minimum=1)
        param_layout.addWidget(self.arrow_density_spin, 6, 1)
        self.arrow_size_label = QLabel("Arrow Size:")
        param_layout.addWidget(self.arrow_size_label, 7, 0)
        self.arrow_size_spin = QSpinBox(value=1, minimum=1)
        param_layout.addWidget(self.arrow_size_spin, 7, 1)
        self.invert_check = QCheckBox("Invert Frames (for dark tracers)")
        param_layout.addWidget(self.invert_check, 8, 0, 1, 2)
        self.bg_subtract_check = QCheckBox("Subtract Background")
        param_layout.addWidget(self.bg_subtract_check, 9, 0, 1, 2)
        self.smooth_check = QCheckBox("Apply Smoothing")
        param_layout.addWidget(self.smooth_check, 10, 0, 1, 2)
        right_splitter.addWidget(self.param_group)

        self.output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(self.output_group)
        self.preview_splitter = QSplitter(Qt.Horizontal)
        self.original_preview = QLabel("Original Frame")
        self.original_preview.setFixedSize(320, 240)
        self.flow_preview = QLabel("Flow Visualization")
        self.flow_preview.setFixedSize(320, 240)
        self.preview_splitter.addWidget(self.original_preview)
        self.preview_splitter.addWidget(self.flow_preview)
        output_layout.addWidget(self.preview_splitter)
        controls_layout = QHBoxLayout()
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.start_processing)
        controls_layout.addWidget(self.process_button)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.export_results)
        controls_layout.addWidget(self.export_button)
        output_layout.addLayout(controls_layout)
        self.progress_bar = QProgressBar()
        output_layout.addWidget(self.progress_bar)
        right_splitter.addWidget(self.output_group)
        main_layout.addWidget(right_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Enable drag-drop on window
        self.setAcceptDrops(True)

        # Initial visibility
        self.update_params_visibility(self.algo_combo.currentText())

    def update_params_visibility(self, algo):
        is_flowtrace = (algo == "FlowTrace")
        self.win_size_label.setText("Trace Length:" if is_flowtrace else "Window Size:")
        self.pyr_levels_label.setVisible(not is_flowtrace)
        self.pyr_levels_spin.setVisible(not is_flowtrace)
        self.iter_label.setVisible(not is_flowtrace)
        self.iter_spin.setVisible(not is_flowtrace)
        self.mag_label.setVisible(not is_flowtrace)
        self.mag_slider.setVisible(not is_flowtrace)
        self.arrow_density_label.setVisible(not is_flowtrace)
        self.arrow_density_spin.setVisible(not is_flowtrace)
        self.arrow_size_label.setVisible(not is_flowtrace)
        self.arrow_size_spin.setVisible(not is_flowtrace)
        self.invert_check.setVisible(is_flowtrace)
        self.bg_subtract_check.setVisible(is_flowtrace)
        # cmap and smooth are shared

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and files[0].lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            self.load_video(files[0])

    def load_video(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            try:
                self.metadata = get_video_metadata(path)
                self.metadata_label.setText(
                    f"Resolution: {self.metadata['resolution']} | FPS: {self.metadata['fps']} | "
                    f"Duration: {self.metadata['duration']:.2f}s | Codec: {self.metadata['codec']}"
                )
                self.cap = cv2.VideoCapture(path, cv2.CAP_MSMF)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.start_spin.setRange(0, self.total_frames - 1)
                self.end_spin.setRange(0, self.total_frames - 1)
                self.end_spin.setValue(self.total_frames - 1)
                self.seek_slider.setRange(0, self.total_frames - 1)
                self.update_preview()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load video: {str(e)}")

    def toggle_playback(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        else:
            self.timer.start(1000 / self.metadata.get('fps', 30))
            self.play_button.setText("Pause")

    def update_preview(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_image(frame, self.preview_label)
                self.seek_slider.setValue(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            else:
                self.timer.stop()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def seek_video(self, position):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            self.update_preview()

    def toggle_entire(self, state):
        self.start_spin.setEnabled(not state)
        self.end_spin.setEnabled(not state)
        if state:
            self.start_spin.setValue(0)
            self.end_spin.setValue(self.total_frames - 1)

    def start_processing(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "No video loaded")
            return
        self.processed_viz = []
        params = {
            'algorithm': self.algo_combo.currentText(),
            'win_size': self.win_size_spin.value(),
            'pyr_levels': self.pyr_levels_spin.value(),
            'iterations': self.iter_spin.value(),
            'mag_threshold': self.mag_slider.value() / 100.0,
            'cmap': self.cmap_combo.currentText(),
            'arrow_density': self.arrow_density_spin.value(),
            'arrow_size': self.arrow_size_spin.value(),
            'smooth': self.smooth_check.isChecked(),
            'start_frame': self.start_spin.value(),
            'end_frame': self.end_spin.value(),
            'step': self.step_spin.value(),
            'invert': self.invert_check.isChecked(),
            'bg_subtract': self.bg_subtract_check.isChecked()
        }
        self.thread = FlowProcessorThread(self.video_path, params)
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.frame_ready.connect(self.update_flow_preview)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.handle_error)
        self.thread.start()
        self.status_bar.showMessage("Processing...")

    def update_flow_preview(self, original, flow_viz):
        self.display_image(original, self.original_preview)
        self.display_image(flow_viz, self.flow_preview)
        self.processed_viz.append(flow_viz.copy())

    def processing_finished(self):
        self.status_bar.showMessage("Processing complete")

    def handle_error(self, msg):
        QMessageBox.warning(self, "Error", msg)
        self.status_bar.showMessage("Processing failed")

    def export_results(self):
        if not self.processed_viz:
            QMessageBox.warning(self, "Error", "No processed results to export")
            return
        export_type, ok = QInputDialog.getItem(self, "Export Type", "Select format:", ["Video (MP4)", "Image Sequence", "Numpy Array"], 0, False)
        if not ok:
            return
        if export_type == "Video (MP4)":
            path, _ = QFileDialog.getSaveFileName(self, "Save Video", "", "Videos (*.mp4)")
            if path:
                if not path.endswith('.mp4'):
                    path += '.mp4'
                height, width = self.processed_viz[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = self.metadata.get('fps', 30)
                out = cv2.VideoWriter(path, fourcc, fps, (width, height))
                for viz in self.processed_viz:
                    out.write(viz)
                out.release()
                QMessageBox.information(self, "Success", "Video exported successfully")
        elif export_type == "Image Sequence":
            dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Images")
            if dir_path:
                for i, viz in enumerate(self.processed_viz):
                    cv2.imwrite(os.path.join(dir_path, f"frame_{i:04d}.png"), viz)
                QMessageBox.information(self, "Success", "Image sequence saved")
        elif export_type == "Numpy Array":
            path, _ = QFileDialog.getSaveFileName(self, "Save Numpy Array", "", "Numpy (*.npy)")
            if path:
                if not path.endswith('.npy'):
                    path += '.npy'
                np.save(path, np.stack(self.processed_viz, axis=0))
                QMessageBox.information(self, "Success", "Numpy array saved")

    def display_image(self, img, label):
        if img is not None:
            height, width = img.shape[:2]
            if len(img.shape) == 3:
                img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                bytes_per_line = 3 * width
                format = QImage.Format_RGB888
            else:
                img_display = img
                bytes_per_line = width
                format = QImage.Format_Grayscale8
            qimg = QImage(img_display.data, width, height, bytes_per_line, format)
            label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio))