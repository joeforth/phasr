import sys
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
                             QComboBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from RR import VideoThread



class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Tracking UI")
        self.setGeometry(100, 100, 1200, 700)

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.video_label = QLabel(self)
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Press 'Start Tracking' to begin")
        self.video_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.video_label, 3)

        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout, 1)
        # CHANGE: Add Camera Selection Dropdown
        controls_layout.addWidget(QLabel("Camera Selection"))
        self.camera_combo = QComboBox()
        self.populate_cameras()
        controls_layout.addWidget(self.camera_combo)

        self.start_stop_button = QPushButton("Start Tracking")
        self.start_stop_button.clicked.connect(self.toggle_video)
        controls_layout.addWidget(self.start_stop_button)

        self.conf_slider = self.create_slider("Confidence", 50, 0, 100, self.update_params)
        controls_layout.addWidget(self.conf_slider['label'])
        controls_layout.addWidget(self.conf_slider['slider'])

        self.iou_slider = self.create_slider("IOU", 10, 0, 100, self.update_params)
        controls_layout.addWidget(self.iou_slider['label'])
        controls_layout.addWidget(self.iou_slider['slider'])

        self.dist_slider = self.create_slider("Distance Threshold", 25, 1, 100, self.update_params)
        controls_layout.addWidget(self.dist_slider['label'])
        controls_layout.addWidget(self.dist_slider['slider'])

        self.smooth_slider = self.create_slider("Smoothing Factor", 50, 0, 100, self.update_params)
        controls_layout.addWidget(self.smooth_slider['label'])
        controls_layout.addWidget(self.smooth_slider['slider'])

        self.live_slider = self.create_slider("max detection count", 25, 1, 100, self.update_params)
        controls_layout.addWidget(self.live_slider['label'])
        controls_layout.addWidget(self.live_slider['slider'])

        controls_layout.addStretch()

        self.info_table = QTableWidget(self)
        self.info_table.setColumnCount(4)
        self.info_table.setHorizontalHeaderLabels(["ID", "Class", "Confidence","Count"])
        self.info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        controls_layout.addWidget(QLabel("Tracked Objects"))
        controls_layout.addWidget(self.info_table)

        self.thread = None
        self.is_running = False
        self.update_params()

    def populate_cameras(self):
        """Detects available cameras using pygrabber and adds them to the dropdown."""
        try:
            from pygrabber.dshow_graph import FilterGraph

            graph = FilterGraph()
            devices = graph.get_input_devices()  # Returns a list of camera names
            self.camera_combo.addItems(devices)

            if not devices:
                self.start_stop_button.setEnabled(False)
                self.video_label.setText("No cameras found!")

        except (ImportError, ModuleNotFoundError):
            # Fallback to the old method if pygrabber is not installed
            self.video_label.setText("PyGrabber not found. Using slower method.")
            self.populate_cameras_fallback()
        except Exception as e:
            # Handle other potential errors with pygrabber
            self.video_label.setText(f"Error detecting cameras: {e}")
            self.start_stop_button.setEnabled(False)
    def create_slider(self, name, value, min_val, max_val, callback):
        label = QLabel(f"{name}: {value / 100:.2f}")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        return {"label": label, "slider": slider}

    def update_params(self):
        conf = self.conf_slider['slider'].value() / 100.0
        iou = self.iou_slider['slider'].value() / 100.0
        dist = self.dist_slider['slider'].value()
        smooth = self.smooth_slider['slider'].value() / 100.0
        liv = self.live_slider['slider'].value()

        self.conf_slider['label'].setText(f"Confidence: {conf:.2f}")
        self.iou_slider['label'].setText(f"IOU: {iou:.2f}")
        self.dist_slider['label'].setText(f"Distance Threshold: {dist}")
        self.smooth_slider['label'].setText(f"Smoothing Factor: {smooth:.2f}")
        self.live_slider['label'].setText(f"Liveliness Factor: {liv}")

        if self.thread and self.is_running:
            self.thread.update_params(conf, iou, dist, smooth, liv)

    def toggle_video(self):
        if not self.is_running:
            self.is_running = True
            # CHANGE: Disable camera dropdown and pass selected index to thread
            self.camera_combo.setEnabled(False)
            camera_index = self.camera_combo.currentIndex()
            self.thread = VideoThread(camera_index)

            self.start_stop_button.setText("Stop Tracking")
            self.update_params()
            self.thread.change_pixmap_signal.connect(self.update_image)
            self.thread.update_table_signal.connect(self.update_table)
            self.thread.start()
        else:
            self.is_running = False
            # CHANGE: Re-enable camera dropdown
            self.camera_combo.setEnabled(True)
            self.start_stop_button.setText("Start Tracking")
            if self.thread:
                self.thread.stop()
                self.thread = None
            self.video_label.setText("Select a camera and press 'Start Tracking'")
            self.info_table.setRowCount(0)

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    @pyqtSlot(list)
    def update_table(self, tracked_objects):
        self.info_table.setRowCount(len(tracked_objects))
        for i, obj in enumerate(tracked_objects):
            self.info_table.setItem(i, 0, QTableWidgetItem(str(obj.id)))
            self.info_table.setItem(i, 1, QTableWidgetItem(str(int(obj.class_id))))
            self.info_table.setItem(i, 2, QTableWidgetItem(f"{obj.conf:.2f}"))
            self.info_table.setItem(i, 3, QTableWidgetItem(f"{obj.liveness_counter:.2f}"))


    def convert_cv_qt(self, cv_img):
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        if self.thread:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
