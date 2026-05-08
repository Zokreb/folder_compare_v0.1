import json
import os
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QCheckBox, 
                             QScrollArea, QProgressBar, QFrame, QLineEdit, QFileDialog,
                             QComboBox, QSlider) # <-- Added QSlider
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
        self.task_func(*self.args, progress_callback=self.progress.emit)
        self.finished.emit()

class FolderInputZone(QWidget):
    folder_dropped = pyqtSignal(str)

    def __init__(self, title):
        super().__init__()
        self.setAcceptDrops(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        lbl_title = QLabel(f"<b>{title}</b>")
        layout.addWidget(lbl_title)
        
        input_layout = QHBoxLayout()
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText("Drag and drop folder here or browse...")
        self.line_edit.setReadOnly(True)  # Keeps the user from typing invalid paths manually
        
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.browse_folder)
        
        input_layout.addWidget(self.line_edit)
        input_layout.addWidget(self.btn_browse)
        
        layout.addLayout(input_layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            self.set_folder(folder)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.set_folder(path)
            
    def set_folder(self, path):
        self.line_edit.setText(path)
        self.folder_dropped.emit(path)

class PairWidget(QWidget):
    def __init__(self, data, core, hash_size=8, parent=None):
        super().__init__(parent)
        self.core = core
        self.data = data
        self.hash_size = hash_size # Store it
        
        # Unpack the data
        if len(self.data) == 11:
            (self.o_path, self.o_name, self.o_size, self.o_w, self.o_h,
             self.c_path, self.c_name, self.c_size, self.c_w, self.c_h,
             self.distance) = self.data
        else:
            (self.o_path, self.o_name, self.o_size, self.o_w, self.o_h,
             self.c_path, self.c_name, self.c_size, self.c_w, self.c_h) = self.data
            self.distance = 10

        layout = QHBoxLayout(self)
        
        # --- 1. CREATE ALL WIDGETS FIRST ---
        
        # Left: Original + Overlay
        self.lbl_orig = QLabel()
        self.lbl_orig.setFixedSize(300, 300)
        self.lbl_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Middle: Info & Actions
        self.mid_widget = QWidget()
        self.mid_widget.setFixedWidth(250)
        self.mid_layout = QVBoxLayout(self.mid_widget)
        
        # Right: Raw Cropped Image
        self.lbl_crop = QLabel()
        self.lbl_crop.setFixedSize(300, 300)
        self.lbl_crop.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- 2. POPULATE THE DATA ---
        # Parse the distance and calculate confidence
        if isinstance(self.distance, bytes):
            self.distance = int.from_bytes(self.distance, byteorder='little')
        else:
            self.distance = float(self.distance)
        
        # Calculate Confidence Math (assuming default hash_size=8, so 32 is 0%)
        # Total bits = hash_size squared. Random noise threshold is half of that.
        max_distance = (self.hash_size ** 2) / 2.0
        
        self.confidence = max(0.0, (1.0 - (self.distance / max_distance)) * 100.0)
        
        lbl_conf = QLabel(f"<b>Match Confidence: {self.confidence:.1f}%</b>")
        if self.confidence > 80:
            lbl_conf.setStyleSheet("color: #2E7D32; font-size: 14px;") # Green
        elif self.confidence > 50:
            lbl_conf.setStyleSheet("color: #F57F17; font-size: 14px;") # Orange
        else:
            lbl_conf.setStyleSheet("color: #C62828; font-size: 14px;") # Red
            
        self.mid_layout.addWidget(lbl_conf)
        self.mid_layout.addWidget(QLabel("---"))
        
        self.mid_layout.addWidget(QLabel(f"<b>Original:</b> {self.o_name}"))
        self.mid_layout.addWidget(QLabel(f"Size: {self.o_size / 1024:.1f} KB | {self.o_w}x{self.o_h}"))
        self.mid_layout.addWidget(QLabel("---"))
        self.mid_layout.addWidget(QLabel(f"<b>Cropped:</b> {self.c_name}"))
        self.mid_layout.addWidget(QLabel(f"Size: {self.c_size / 1024:.1f} KB | {self.c_w}x{self.c_h}"))
        
        # Calculate and Display % Area Lost
        if self.o_w * self.o_h > 0:
            self.crop_loss_pct = (1.0 - ((self.c_w * self.c_h) / (self.o_w * self.o_h))) * 100
        else:
            self.crop_loss_pct = 0.0
            
        self.mid_layout.addWidget(QLabel(f"<b>Area Lost:</b> {self.crop_loss_pct:.1f}%"))
        
        # EXACT Math Warning Check
        if self.c_w > self.o_w or self.c_h > self.o_h or (self.c_w * self.c_h) > (self.o_w * self.o_h):
            lbl_warn = QLabel("<b>⚠️ WARNING: Cropped image has larger dimensions!</b>")
            lbl_warn.setStyleSheet("color: #D84315; background-color: #FBE9E7; padding: 4px; border-radius: 4px; border: 1px solid #D84315;")
            lbl_warn.setWordWrap(True)
            self.mid_layout.addWidget(lbl_warn)

        self.btn_replace = QPushButton("Replace Single")
        self.btn_replace.clicked.connect(self.replace_single)
        self.chk_mass = QCheckBox("Select")
        
        self.mid_layout.addWidget(self.btn_replace)
        self.mid_layout.addWidget(self.chk_mass)

        # --- 3. BUILD LAYOUT AND LOAD IMAGES ---
        
        layout.addWidget(self.lbl_orig)
        layout.addWidget(self.mid_widget)
        layout.addWidget(self.lbl_crop)
        
        # Safely load the images now that ALL labels exist in memory
        self.load_overlay()
        self.load_image(self.c_path, self.lbl_crop)

    def load_image(self, path, label):
        pixmap = QPixmap(path)
        label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def load_overlay(self):
        img_rgb = self.core.generate_overlay(self.o_path, self.c_path)
        
        if img_rgb is not None:
            # We must convert back to BGR just for the imencode step
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            # Encode the image to a memory buffer (simulating a saved .jpg file)
            is_success, buffer = cv2.imencode(".jpg", img_bgr)
            
            if is_success:
                pixmap = QPixmap()
                # Load the pixmap directly from the raw bytes
                pixmap.loadFromData(buffer.tobytes())
                
                self.lbl_orig.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                return # Exit the function successfully
                
        # Fallback if homography or encoding fails
        self.load_image(self.o_path, self.lbl_orig)

    def replace_single(self):
        self.core.replace_image(self.o_path, self.c_path)
        self.btn_replace.setText("Replaced!")
        self.btn_replace.setEnabled(False)
        self.chk_mass.setChecked(False)
        self.chk_mass.setEnabled(False)

    def load_image(self, path, label):
        pixmap = QPixmap(path)
        label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def load_overlay(self):
        # Now targets self.lbl_orig instead of self.lbl_crop
        img_rgb = self.core.generate_overlay(self.o_path, self.c_path)
        if img_rgb is not None:
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_orig.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            # Fallback to the raw original if OpenCV fails to find homography
            self.load_image(self.o_path, self.lbl_orig)

    def replace_single(self):
        self.core.replace_image(self.o_path, self.c_path)
        self.btn_replace.setText("Replaced!")
        self.btn_replace.setEnabled(False)
        self.chk_mass.setChecked(False)
        self.chk_mass.setEnabled(False)

    def load_image(self, path, label):
        pixmap = QPixmap(path)
        label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    """def load_overlay(self):
        img_rgb = self.core.generate_overlay(self.o_path, self.c_path)
        if img_rgb is not None:
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            self.lbl_crop.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.load_image(self.c_path, self.lbl_crop)"""

    def replace_single(self):
        self.core.replace_image(self.o_path, self.c_path)
        self.btn_replace.setText("Replaced!")
        self.btn_replace.setEnabled(False)
        self.chk_mass.setChecked(False)
        self.chk_mass.setEnabled(False)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Match & Restore")
        self.setGeometry(100, 100, 1100, 800)
        self.core = ImageProcessor()

        # --- NEW: Load Settings & DB Failsafe ---
        self.settings = self.load_settings()
        
        # Determine the true state of the DB
        db_hash = self.core.get_existing_hash_size()
        if db_hash is not None:
            self.current_db_hash_size = db_hash
            # Force the slider to respect the actual DB state if JSON was deleted/desynced
            self.settings["hash_size"] = db_hash 
        else:
            self.current_db_hash_size = self.settings["hash_size"]            
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)

        # Top Controls: Folder Inputs
        drop_layout = QHBoxLayout()
        self.drop_orig = FolderInputZone("Original Images")
        self.drop_orig.line_edit.setText(self.settings["orig_folder"]) # Load saved path
        self.drop_orig.folder_dropped.connect(lambda path: self.run_scan(path, "original"))
        
        self.drop_crop = FolderInputZone("Cropped Images")
        self.drop_crop.line_edit.setText(self.settings["crop_folder"]) # Load saved path
        self.drop_crop.folder_dropped.connect(lambda path: self.run_scan(path, "cropped"))
        
        drop_layout.addWidget(self.drop_orig)
        drop_layout.addWidget(self.drop_crop)
        self.main_layout.addLayout(drop_layout)

        # Top Controls: Actions
        action_layout = QHBoxLayout()
        self.chk_force_rebuild = QCheckBox("Force Recache (Ignore DB)")
        self.btn_process = QPushButton("Find Matches")
        self.btn_process.clicked.connect(self.start_matching)
        
        self.btn_toggle_sel = QPushButton("Toggle Select All")
        self.btn_toggle_sel.clicked.connect(self.toggle_selection)
        self.toggle_state = False
        
        self.btn_batch = QPushButton("Mass Replace Selected")
        self.btn_batch.clicked.connect(self.batch_replace)

        #Sort controls
        self.combo_sort = QComboBox()
        self.combo_sort.addItems([
            "Sort: Default (Match Quality)",
            "Sort: % Area Lost (Smallest First)",
            "Sort: % Area Lost (Largest First)",
            "Sort: Cropped Width (Largest First)",
            "Sort: Cropped Height (Largest First)",
            "Sort: Confidence (Highest First)",
            "Sort: Confidence (Lowest First)"           
        ])
        self.combo_sort.currentIndexChanged.connect(self.sort_widgets)
        self.combo_sort.setEnabled(False) # Disabled until matching is done

        # --- NEW: Settings / Sliders Panel ---
        settings_layout = QHBoxLayout()
        
        h_val = self.settings["hash_size"]
        self.lbl_hash = QLabel(f"<b>Hash Size:</b> {h_val}")
        self.slider_hash = QSlider(Qt.Orientation.Horizontal)
        self.slider_hash.setRange(4, 24)
        self.slider_hash.setSingleStep(2)
        self.slider_hash.setValue(h_val) # Apply saved value
        self.slider_hash.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_hash.setTickInterval(4)
        self.slider_hash.valueChanged.connect(self.on_hash_changed)
        
        t_val = self.settings["threshold"]
        self.lbl_thresh = QLabel(f"<b>Match Threshold:</b> {t_val}")
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(0, 60)
        self.slider_thresh.setValue(t_val) # Apply saved value
        self.slider_thresh.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_thresh.setTickInterval(5)
        self.slider_thresh.valueChanged.connect(lambda v: self.lbl_thresh.setText(f"<b>Match Threshold:</b> {v}"))

        settings_layout.addWidget(self.lbl_hash)
        settings_layout.addWidget(self.slider_hash)
        settings_layout.addWidget(QLabel("   |   ")) # Spacer
        settings_layout.addWidget(self.lbl_thresh)
        settings_layout.addWidget(self.slider_thresh)
        self.main_layout.addLayout(settings_layout)
        
        action_layout.addWidget(self.chk_force_rebuild)
        action_layout.addWidget(self.btn_process)
        action_layout.addWidget(self.combo_sort) # Added to layout
        action_layout.addWidget(self.btn_toggle_sel)
        action_layout.addWidget(self.btn_batch)
        self.main_layout.addLayout(action_layout)

        # Status and Progress
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("Ready.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.progress_bar)
        self.main_layout.addLayout(status_layout)

        # Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

        self.pair_widgets = []

    def load_settings(self):
        # Default fallback settings
        settings = {
            "hash_size": 8,
            "threshold": 10,
            "orig_folder": "",
            "crop_folder": ""
        }
        
        if os.path.exists("settings.json"):
            try:
                with open("settings.json", "r") as f:
                    loaded = json.load(f)
                    settings.update(loaded) # Overwrite defaults with saved values
            except Exception as e:
                print(f"Error loading settings: {e}")
                
        return settings

    def save_settings(self):
        settings = {
            "hash_size": self.slider_hash.value(),
            "threshold": self.slider_thresh.value(),
            "orig_folder": self.drop_orig.line_edit.text(),
            "crop_folder": self.drop_crop.line_edit.text()
        }
        try:
            with open("settings.json", "w") as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def closeEvent(self, event):
        """This built-in PyQt method triggers automatically when you X out of the window."""
        self.save_settings()
        event.accept()

    def run_scan(self, path, folder_type):
        self.lbl_status.setText(f"Scanning {folder_type} folder...")
        self.progress_bar.setValue(0)
        force = self.chk_force_rebuild.isChecked()
        h_size = self.slider_hash.value() # <-- Grab the slider value
        
        # Pass h_size to the worker
        self.worker = WorkerThread(self.core.scan_folder, path, folder_type, force, h_size)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(lambda: self.lbl_status.setText(f"Finished caching {folder_type}."))
        self.worker.start()

    def clear_layout(self):
        for i in reversed(range(self.scroll_layout.count())): 
            widgetToRemove = self.scroll_layout.itemAt(i).widget()
            self.scroll_layout.removeWidget(widgetToRemove)
            widgetToRemove.setParent(None)
        self.pair_widgets.clear()

    def start_matching(self):
        orig_path = self.drop_orig.line_edit.text()
        crop_path = self.drop_crop.line_edit.text()
        
        if not orig_path or not crop_path:
            self.lbl_status.setText("Please load both Original and Cropped folders first.")
            return

        self.clear_layout()
        
        # Check if the slider changed OR the force rebuild box is ticked
        if self.slider_hash.value() != getattr(self, 'current_db_hash_size', 8) or self.chk_force_rebuild.isChecked():
            self.lbl_status.setText("Recaching required. Scanning Original folder...")
            self.progress_bar.setValue(0)
            
            # Lock UI to prevent the user from clicking around during the chain
            self.btn_process.setEnabled(False)
            self.btn_batch.setEnabled(False)
            self.combo_sort.setEnabled(False)
            
            h_size = self.slider_hash.value()
            
            # Step 1: Start Original Scan
            self.scan_worker_orig = WorkerThread(self.core.scan_folder, orig_path, "original", True, h_size)
            self.scan_worker_orig.progress.connect(self.progress_bar.setValue)
            self.scan_worker_orig.finished.connect(self._chain_scan_cropped) # Move to Step 2 when done
            self.scan_worker_orig.start()
        else:
            # No recache needed, go straight to matching
            self._chain_matching()
            
    def _chain_scan_cropped(self):
        self.lbl_status.setText("Scanning Cropped folder...")
        self.progress_bar.setValue(0)
        
        crop_path = self.drop_crop.line_edit.text()
        h_size = self.slider_hash.value()
        
        # Step 2: Start Cropped Scan
        self.scan_worker_crop = WorkerThread(self.core.scan_folder, crop_path, "cropped", True, h_size)
        self.scan_worker_crop.progress.connect(self.progress_bar.setValue)
        self.scan_worker_crop.finished.connect(self._chain_matching) # Move to Step 3 when done
        self.scan_worker_crop.start()

    def _chain_matching(self):
        # Update our tracking variable and uncheck the rebuild box
        self.current_db_hash_size = self.slider_hash.value()
        self.chk_force_rebuild.setChecked(False)
        
        self.lbl_status.setText("Matching images...")
        self.progress_bar.setValue(0)
        
        # Lock UI in case we skipped straight to this step
        self.btn_process.setEnabled(False)
        
        thresh = self.slider_thresh.value()
        
        # Step 3: Find Matches
        self.match_worker = WorkerThread(self.core.find_matches, thresh)
        self.match_worker.progress.connect(self.progress_bar.setValue)
        self.match_worker.finished.connect(self.on_matching_finished) # Render widgets when done
        self.match_worker.start()            

    def on_matching_finished(self):
        matches = self.core.get_match_pairs()
        total = len(matches)
        
        if total == 0:
            self.lbl_status.setText("No matches found.")
            return

        self.lbl_status.setText(f"Generating {total} previews and overlays...")
        self.progress_bar.setValue(0)
        
        self.btn_process.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.chk_force_rebuild.setEnabled(False)
        self.combo_sort.setEnabled(False)
        
        for i, match in enumerate(matches):
            # --- UPDATE 3: Pass the dynamic hash size into the widget ---
            pw = PairWidget(match, self.core, self.current_db_hash_size) 
            
            pw.default_index = i
            self.scroll_layout.addWidget(pw)
            self.pair_widgets.append(pw)
            
            self.progress_bar.setValue(int((i + 1) / total * 100))
            QApplication.processEvents()
            
        self.lbl_status.setText(f"Found {total} matches. Ready.")
        
        self.btn_process.setEnabled(True)
        self.btn_batch.setEnabled(True)
        self.chk_force_rebuild.setEnabled(True)
        self.combo_sort.setEnabled(True) # <-- Enable sorting

    def batch_replace(self):
        selected = [pw for pw in self.pair_widgets if pw.chk_mass.isChecked() and pw.btn_replace.isEnabled()]
        total = len(selected)
        if total == 0:
            return
            
        self.lbl_status.setText(f"Running mass replacement for {total} files...")
        self.progress_bar.setValue(0)
        
        self.btn_process.setEnabled(False)
        self.btn_batch.setEnabled(False)

        for i, pw in enumerate(selected):
            pw.replace_single()
            self.progress_bar.setValue(int((i + 1) / total * 100))
            
            # Keep the UI responsive during heavy disk I/O
            QApplication.processEvents()
            
        self.lbl_status.setText("Batch replacement complete.")
        self.btn_process.setEnabled(True)
        self.btn_batch.setEnabled(True)

    def on_hash_changed(self, value):
        self.lbl_hash.setText(f"<b>Hash Size:</b> {value}")
        self.chk_force_rebuild.setChecked(True)
        self.lbl_status.setText("Hash size changed. 'Force Recache' automatically selected.")

    def toggle_selection(self):
        self.toggle_state = not self.toggle_state
        for pw in self.pair_widgets:
            if pw.btn_replace.isEnabled():
                pw.chk_mass.setChecked(self.toggle_state)

    def batch_replace(self):
        selected = [pw for pw in self.pair_widgets if pw.chk_mass.isChecked() and pw.btn_replace.isEnabled()]
        total = len(selected)
        if total == 0:
            return
            
        self.lbl_status.setText("Running mass replacement...")
        for i, pw in enumerate(selected):
            pw.replace_single()
            self.progress_bar.setValue(int((i + 1) / total * 100))
            
        self.lbl_status.setText("Batch replacement complete.")

    def sort_widgets(self):
        if not self.pair_widgets:
            return

        sort_type = self.combo_sort.currentText()
        
        # Sort the python list based on widget attributes
        if sort_type == "Sort: % Area Lost (Smallest First)":
            self.pair_widgets.sort(key=lambda pw: pw.crop_loss_pct)
        elif sort_type == "Sort: % Area Lost (Largest First)":
            self.pair_widgets.sort(key=lambda pw: pw.crop_loss_pct, reverse=True)
        elif sort_type == "Sort: Cropped Width (Largest First)":
            self.pair_widgets.sort(key=lambda pw: pw.c_w, reverse=True)
        elif sort_type == "Sort: Cropped Height (Largest First)":
            self.pair_widgets.sort(key=lambda pw: pw.c_h, reverse=True)
        elif sort_type == "Sort: Confidence (Highest First)":
            self.pair_widgets.sort(key=lambda pw: pw.confidence, reverse=True)
        elif sort_type == "Sort: Confidence (Lowest First)":
            self.pair_widgets.sort(key=lambda pw: pw.confidence)            
        else:
            # Default fallback using the index we saved during generation
            self.pair_widgets.sort(key=lambda pw: pw.default_index)

        # Re-insert widgets into the layout in the new order.
        for i, pw in enumerate(self.pair_widgets):
            self.scroll_layout.insertWidget(i, pw)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())