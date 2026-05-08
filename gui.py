import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QCheckBox, 
                             QScrollArea, QProgressBar, QFileDialog)
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import cv2

from core import ImageProcessor

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, task_func, *args):
        super().__init__()
        self.task_func = task_func
        self.args = args

    def run(self):
        self.task_func(*self.args, self.progress.emit)
        self.finished.emit()

class PairWidget(QWidget):
    def __init__(self, data, core, parent=None):
        super().__init__(parent)
        self.core = core
        self.data = data
        (self.o_path, self.o_name, self.o_size, self.o_w, self.o_h,
         self.c_path, self.c_name, self.c_size, self.c_w, self.c_h) = data

        layout = QHBoxLayout(self)
        
        # Left: Original
        self.lbl_orig = QLabel()
        self.lbl_orig.setFixedSize(300, 300)
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_image(self.o_path, self.lbl_orig)
        
        # Middle: Info & Actions
        mid_layout = QVBoxLayout()
        mid_layout.addWidget(QLabel(f"<b>Original:</b> {self.o_name}"))
        mid_layout.addWidget(QLabel(f"Size: {self.o_size / 1024:.1f} KB | {self.o_w}x{self.o_h}"))
        mid_layout.addWidget(QLabel("---"))
        mid_layout.addWidget(QLabel(f"<b>Cropped:</b> {self.c_name}"))
        mid_layout.addWidget(QLabel(f"Size: {self.c_size / 1024:.1f} KB | {self.c_w}x{self.c_h}"))
        
        self.btn_replace = QPushButton("Replace Single")
        self.btn_replace.clicked.connect(self.replace_single)
        self.chk_mass = QCheckBox("Select for Mass Replace")
        
        mid_layout.addWidget(self.btn_replace)
        mid_layout.addWidget(self.chk_mass)
        mid_widget = QWidget()
        mid_widget.setLayout(mid_layout)
        mid_widget.setFixedWidth(250)
        
        # Right: Cropped (with overlay applied to original preview)
        self.lbl_crop = QLabel()
        self.lbl_crop.setFixedSize(300, 300)
        self.lbl_crop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.load_overlay()

        layout.addWidget(self.lbl_orig)
        layout.addWidget(mid_widget)
        layout.addWidget(self.lbl_crop)

    def load_image(self, path, label):
        pixmap = QPixmap(path)
        label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def load_overlay(self):
        # Generates the ORB overlay box
        img_rgb = self.core.generate_overlay(self.o_path, self.c_path)
        if img_rgb is not None:
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_crop.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.load_image(self.c_path, self.lbl_crop)

    def replace_single(self):
        self.core.replace_image(self.o_path, self.c_path)
        self.btn_replace.setText("Replaced!")
        self.btn_replace.setEnabled(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Match & Restore")
        self.setGeometry(100, 100, 1000, 800)
        self.core = ImageProcessor()
        
        self.setAcceptDrops(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)

        # Top controls
        top_layout = QHBoxLayout()
        self.lbl_status = QLabel("Drag & Drop Original and Cropped folders anywhere.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.btn_process = QPushButton("Find Matches")
        self.btn_process.clicked.connect(self.start_matching)
        self.btn_batch = QPushButton("Mass Replace Selected")
        self.btn_batch.clicked.connect(self.batch_replace)
        
        top_layout.addWidget(self.lbl_status)
        top_layout.addWidget(self.progress_bar)
        top_layout.addWidget(self.btn_process)
        top_layout.addWidget(self.btn_batch)
        self.main_layout.addLayout(top_layout)

        # Scroll Area for matched pairs
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        self.orig_folder = None
        self.crop_folder = None
        self.pair_widgets = []

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            folder_type = "original" if not self.orig_folder else "cropped"
            
            if not self.orig_folder:
                self.orig_folder = path
                self.lbl_status.setText(f"Original set. Drop Cropped folder now.")
                self.run_scan(path, "original")
            elif not self.crop_folder:
                self.crop_folder = path
                self.lbl_status.setText(f"Scanning Cropped...")
                self.run_scan(path, "cropped")

    def run_scan(self, path, folder_type):
        self.worker = WorkerThread(self.core.scan_folder, path, folder_type)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda: self.lbl_status.setText(f"Finished scanning {folder_type}."))
        self.worker.start()

    def start_matching(self):
        self.lbl_status.setText("Matching images...")
        self.core.find_matches()
        matches = self.core.get_match_pairs()
        
        for match in matches:
            pw = PairWidget(match, self.core)
            self.scroll_layout.addWidget(pw)
            self.pair_widgets.append(pw)
            
        self.lbl_status.setText(f"Found {len(matches)} matches.")

    def batch_replace(self):
        selected = [pw for pw in self.pair_widgets if pw.chk_mass.isChecked() and pw.btn_replace.isEnabled()]
        total = len(selected)
        
        for i, pw in enumerate(selected):
            pw.replace_single()
            self.progress_bar.setValue(int((i + 1) / total * 100))
            
        self.lbl_status.setText("Batch replacement complete.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())