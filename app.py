import sys
import os
import cv2
import numpy as np
import tempfile
import io
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QSlider,
    QComboBox, QSpinBox, QGroupBox, QRadioButton, QButtonGroup,
    QScrollArea
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw, ImageFont
import PyPDF2
# Removed pdf2image dependency - using pure Python approach instead

def is_sift_available():
    """
    Check if SIFT is available (i.e., opencv-contrib is installed).
    We'll attempt to load cv2.xfeatures2d.SIFT_create.
    Returns True if available, False if not.
    """
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        return True
    except AttributeError:
        return False
    except cv2.error:
        return False

class ImageDiffApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image & PDF Difference Viewer")
        self.setGeometry(200, 200, 1300, 700)

        # Main container widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Overall layout
        self.layout = QVBoxLayout()
        
        # Top: Buttons + parameter sliders
        self.top_layout = QVBoxLayout()  # Changed to vertical layout
        
        # Create horizontal layouts for button groups
        self.image_buttons_layout = QHBoxLayout()
        self.action_buttons_layout = QHBoxLayout()
        self.parameter_layout = QHBoxLayout()
        
        # Style for button groups
        button_style = """
            QPushButton {
                padding: 8px 15px;
                font-size: 12px;
                border: 1px solid #ccc;
                border-radius: 4px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
        """
        
        # Group 1: Image/PDF Loading Buttons
        self.load_before_btn = QPushButton("Load 'Before' File")
        self.load_after_btn = QPushButton("Load 'After' File")
        self.clear_before_btn = QPushButton("Clear Before")
        self.clear_after_btn = QPushButton("Clear After")
        self.clear_all_btn = QPushButton("Clear All")
        
        # Group 2: Action Buttons
        self.compare_btn = QPushButton("Compare Images")
        self.save_result_btn = QPushButton("Save Result")
        
        # Apply styles
        for btn in [self.load_before_btn, self.load_after_btn, self.clear_before_btn, 
                   self.clear_after_btn, self.clear_all_btn, self.compare_btn, 
                   self.save_result_btn]:
            btn.setStyleSheet(button_style)
        
        # Special styling for main action buttons
        action_button_style = """
            QPushButton {
                padding: 8px 20px;
                font-size: 13px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """
        self.compare_btn.setStyleSheet(action_button_style)
        self.save_result_btn.setStyleSheet(action_button_style)
        
        # Add buttons to their respective layouts
        self.image_buttons_layout.addWidget(self.load_before_btn)
        self.image_buttons_layout.addWidget(self.clear_before_btn)
        self.image_buttons_layout.addWidget(self.load_after_btn)
        self.image_buttons_layout.addWidget(self.clear_after_btn)
        self.image_buttons_layout.addWidget(self.clear_all_btn)
        
        self.action_buttons_layout.addStretch()
        self.action_buttons_layout.addWidget(self.compare_btn)
        self.action_buttons_layout.addWidget(self.save_result_btn)
        self.action_buttons_layout.addStretch()

        # Slider styling
        slider_style = """
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #ffffff;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """
        
        # Parameter controls
        parameter_group = QWidget()
        parameter_group.setStyleSheet("""
            QLabel {
                font-size: 12px;
                margin-right: 10px;
            }
        """)
        
        self.threshold_label = QLabel("Threshold: 50")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.setStyleSheet(slider_style)
        
        self.kernel_label = QLabel("Kernel Size: 5")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(5)
        self.kernel_slider.setStyleSheet(slider_style)
        
        # Add sliders/labels to parameter layout
        self.parameter_layout.addWidget(self.threshold_label)
        self.parameter_layout.addWidget(self.threshold_slider)
        self.parameter_layout.addWidget(self.kernel_label)
        self.parameter_layout.addWidget(self.kernel_slider)

        # PDF Navigation Controls
        self.pdf_controls_layout = QHBoxLayout()
        
        # File type selection
        self.file_type_group = QGroupBox("File Type")
        self.file_type_layout = QHBoxLayout()
        self.image_radio = QRadioButton("Image")
        self.pdf_radio = QRadioButton("PDF")
        self.image_radio.setChecked(True)  # Default to image mode
        self.file_type_layout.addWidget(self.image_radio)
        self.file_type_layout.addWidget(self.pdf_radio)
        self.file_type_group.setLayout(self.file_type_layout)
        
        # Page navigation for PDFs
        self.page_nav_group = QGroupBox("PDF Page Navigation")
        self.page_nav_layout = QHBoxLayout()
        
        self.before_page_label = QLabel("Before Page:")
        self.before_page_spinner = QSpinBox()
        self.before_page_spinner.setMinimum(1)
        self.before_page_spinner.setEnabled(False)
        
        self.after_page_label = QLabel("After Page:")
        self.after_page_spinner = QSpinBox()
        self.after_page_spinner.setMinimum(1)
        self.after_page_spinner.setEnabled(False)
        
        self.page_nav_layout.addWidget(self.before_page_label)
        self.page_nav_layout.addWidget(self.before_page_spinner)
        self.page_nav_layout.addWidget(self.after_page_label)
        self.page_nav_layout.addWidget(self.after_page_spinner)
        self.page_nav_group.setLayout(self.page_nav_layout)
        
        # Add to PDF controls layout
        self.pdf_controls_layout.addWidget(self.file_type_group)
        self.pdf_controls_layout.addWidget(self.page_nav_group)
        
        # Add all layouts to top layout
        self.top_layout.addLayout(self.image_buttons_layout)
        self.top_layout.addLayout(self.action_buttons_layout)
        self.top_layout.addLayout(self.pdf_controls_layout)
        self.top_layout.addLayout(self.parameter_layout)

        # Middle: Three image labels
        self.images_layout = QHBoxLayout()
        
        # === Image Labels ===
        label_style = """
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f5f5f5;
                padding: 5px;
                font-size: 12px;
            }
        """
        
        # Create scroll areas for zooming
        self.before_scroll = QScrollArea()
        self.after_scroll = QScrollArea()
        self.diff_scroll = QScrollArea()
        
        for scroll in [self.before_scroll, self.after_scroll, self.diff_scroll]:
            scroll.setWidgetResizable(True)
            scroll.setMinimumSize(300, 300)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.before_label = QLabel("Before Image")
        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setMinimumSize(300, 300)
        self.before_label.setStyleSheet(label_style)
        
        self.after_label = QLabel("After Image")
        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setMinimumSize(300, 300)
        self.after_label.setStyleSheet(label_style)
        
        self.diff_label = QLabel("Difference Highlight")
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setMinimumSize(300, 300)
        self.diff_label.setStyleSheet(label_style)
        
        # Set labels as widgets for scroll areas
        self.before_scroll.setWidget(self.before_label)
        self.after_scroll.setWidget(self.after_label)
        self.diff_scroll.setWidget(self.diff_label)
        
        # Initialize zoom scales
        self.zoom_scales = {"before": 1.0, "after": 1.0, "diff": 1.0}
        self.diff_label.setStyleSheet(label_style)

        self.images_layout.addWidget(self.before_scroll)
        self.images_layout.addWidget(self.after_scroll)
        self.images_layout.addWidget(self.diff_scroll)

        # === Bottom layout: difference score display ===
        self.diff_score_label = QLabel("Difference Score: N/A")
        self.bottom_layout = QHBoxLayout()
        self.bottom_layout.addWidget(self.diff_score_label)

        # Combine layouts
        self.layout.addLayout(self.top_layout)
        self.layout.addLayout(self.images_layout)
        self.layout.addLayout(self.bottom_layout)
        self.main_widget.setLayout(self.layout)

        # Store file paths and PDF data
        self.before_image_path = None
        self.after_image_path = None
        self.before_is_pdf = False
        self.after_is_pdf = False
        self.before_pdf_pages = []
        self.after_pdf_pages = []
        self.before_pdf_page_count = 0
        self.after_pdf_page_count = 0
        self.before_temp_dir = None
        self.after_temp_dir = None
        
        # Keep track of the latest difference image for saving
        self.latest_diff_image = None

        # Check if SIFT is available
        self.sift_enabled = is_sift_available()

        # Connect button signals to slots
        self.load_before_btn.clicked.connect(self.load_before_file)
        self.load_after_btn.clicked.connect(self.load_after_file)
        self.clear_before_btn.clicked.connect(self.clear_before_image)
        self.clear_after_btn.clicked.connect(self.clear_after_image)
        self.clear_all_btn.clicked.connect(self.clear_all)
        self.compare_btn.clicked.connect(self.compare_images)
        self.save_result_btn.clicked.connect(self.save_diff_image)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)
        self.kernel_slider.valueChanged.connect(self.update_kernel_label)
        
        # Connect PDF-specific controls
        self.image_radio.toggled.connect(self.toggle_file_type)
        self.pdf_radio.toggled.connect(self.toggle_file_type)
        self.before_page_spinner.valueChanged.connect(self.update_before_page)
        self.after_page_spinner.valueChanged.connect(self.update_after_page)
        
        # Install event filters for wheel events
        self.before_scroll.viewport().installEventFilter(self)
        self.after_scroll.viewport().installEventFilter(self)
        self.diff_scroll.viewport().installEventFilter(self)

    # ------------------- UI Update Methods -------------------
    def update_threshold_label(self, value):
        self.threshold_label.setText(f"Threshold: {value}")

    def update_kernel_label(self, value):
        self.kernel_label.setText(f"Kernel Size: {value}")
        
    def eventFilter(self, obj, event):
        """Handle mouse wheel events for zooming"""
        if event.type() == QEvent.Wheel:
            if obj in [self.before_scroll.viewport(), self.after_scroll.viewport(), self.diff_scroll.viewport()]:
                if event.modifiers() == Qt.ControlModifier:
                    # Determine which scroll area we're in
                    label_type = "before" if obj == self.before_scroll.viewport() else \
                                "after" if obj == self.after_scroll.viewport() else "diff"
                    
                    # Calculate zoom factor
                    delta = event.angleDelta().y()
                    zoom_factor = 1.1 if delta > 0 else 0.9
                    
                    # Update zoom scale
                    self.zoom_scales[label_type] *= zoom_factor
                    self.zoom_scales[label_type] = max(0.1, min(5.0, self.zoom_scales[label_type]))
                    
                    # Refresh the image
                    if label_type == "before":
                        if self.before_is_pdf:
                            self.update_before_page(self.before_page_spinner.value())
                        elif self.before_image_path:
                            self.show_image_in_label(self.before_image_path, self.before_label)
                    elif label_type == "after":
                        if self.after_is_pdf:
                            self.update_after_page(self.after_page_spinner.value())
                        elif self.after_image_path:
                            self.show_image_in_label(self.after_image_path, self.after_label)
                    else:  # diff
                        if self.latest_diff_image is not None:
                            self.show_diff_image(self.latest_diff_image)
                    
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    # ------------------- UI Control Methods -------------------
    def toggle_file_type(self):
        is_pdf_mode = self.pdf_radio.isChecked()
        self.page_nav_group.setEnabled(is_pdf_mode)
        
        # Update file dialog filter based on selected mode
        if is_pdf_mode:
            self.load_before_btn.setText("Load 'Before' PDF")
            self.load_after_btn.setText("Load 'After' PDF")
        else:
            self.load_before_btn.setText("Load 'Before' Image")
            self.load_after_btn.setText("Load 'After' Image")
            
        # Clear any loaded files when switching modes
        self.clear_all()
    
    def update_before_page(self, page_num):
        if self.before_is_pdf and self.before_pdf_pages:
            # Pages are 0-indexed in the list but 1-indexed in the UI
            self.show_pdf_page(page_num - 1, self.before_pdf_pages, self.before_label)
            # Synchronize the after page if it's also a PDF
            if self.after_is_pdf and self.after_pdf_pages:
                # Ensure we don't exceed the page count of the after PDF
                target_page = min(page_num, self.after_pdf_page_count)
                if target_page != self.after_page_spinner.value():
                    self.after_page_spinner.setValue(target_page)
    
    def update_after_page(self, page_num):
        if self.after_is_pdf and self.after_pdf_pages:
            # Pages are 0-indexed in the list but 1-indexed in the UI
            self.show_pdf_page(page_num - 1, self.after_pdf_pages, self.after_label)
            # Synchronize the before page if it's also a PDF
            if self.before_is_pdf and self.before_pdf_pages:
                # Ensure we don't exceed the page count of the before PDF
                target_page = min(page_num, self.before_pdf_page_count)
                if target_page != self.before_page_spinner.value():
                    self.before_page_spinner.setValue(target_page)
    
    # ------------------- Load Files (Images or PDFs) -------------------
    def load_before_file(self):
        if self.pdf_radio.isChecked():
            self.load_pdf_file("before")
        else:
            self.load_image_file("before")
    
    def load_after_file(self):
        if self.pdf_radio.isChecked():
            self.load_pdf_file("after")
        else:
            self.load_image_file("after")
    
    def load_image_file(self, which):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {which.title()} Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)",
            options=options,
        )
        if file_name:
            if which == "before":
                self.before_image_path = file_name
                self.before_is_pdf = False
                self.show_image_in_label(file_name, self.before_label)
            else:  # after
                self.after_image_path = file_name
                self.after_is_pdf = False
                self.show_image_in_label(file_name, self.after_label)
    
    def load_pdf_file(self, which):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {which.title()} PDF",
            "",
            "PDF Files (*.pdf)",
            options=options,
        )
        if file_name:
            try:
                # Create temp directory for PDF page images if needed
                if which == "before":
                    if self.before_temp_dir:
                        # Clean up previous temp directory
                        for file in os.listdir(self.before_temp_dir):
                            os.remove(os.path.join(self.before_temp_dir, file))
                        os.rmdir(self.before_temp_dir)
                    self.before_temp_dir = tempfile.mkdtemp()
                    self.before_image_path = file_name
                    self.before_is_pdf = True
                else:  # after
                    if self.after_temp_dir:
                        # Clean up previous temp directory
                        for file in os.listdir(self.after_temp_dir):
                            os.remove(os.path.join(self.after_temp_dir, file))
                        os.rmdir(self.after_temp_dir)
                    self.after_temp_dir = tempfile.mkdtemp()
                    self.after_image_path = file_name
                    self.after_is_pdf = True
                
                # Process the PDF file
                self.process_pdf_file(file_name, which)                
            except Exception as e:
                QMessageBox.warning(self, "PDF Load Error", str(e))

    def show_image_in_label(self, image_path, label_widget):
        """
        Helper method to load an image using Pillow,
        convert to QPixmap, and display in the given QLabel with zoom support.
        """
        try:
            img = Image.open(image_path)
            qimg = self.pil_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)
            
            # Get the label type (before, after, diff)
            label_type = "before" if label_widget == self.before_label else "after" if label_widget == self.after_label else "diff"
            scale = self.zoom_scales[label_type]
            
            # Calculate scaled size
            scaled_size = pixmap.size() * scale
            label_widget.setPixmap(pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label_widget.setFixedSize(scaled_size)
        except Exception as e:
            QMessageBox.warning(self, "Image Load Error", str(e))

    # ------------------- Compare Images -------------------
    def compare_images(self):
        """
        Align the 'After' image to the 'Before' image (ORB → fallback to SIFT),
        then do an enhanced difference detection in Lab color space,
        using the user-chosen threshold and kernel size.
        Works with both regular images and PDF pages.
        """
        if not self.before_image_path or not self.after_image_path:
            QMessageBox.warning(self, "Missing Files", "Please load both 'Before' and 'After' files first.")
            return

        # Get the images to compare based on file type
        if self.before_is_pdf and self.after_is_pdf:
            # For PDFs, use the currently selected pages
            before_page_idx = self.before_page_spinner.value() - 1  # Convert from 1-indexed to 0-indexed
            after_page_idx = self.after_page_spinner.value() - 1
            
            if before_page_idx >= len(self.before_pdf_pages) or after_page_idx >= len(self.after_pdf_pages):
                QMessageBox.warning(self, "Invalid Page", "Selected page number is out of range.")
                return
                
            # Convert PIL images to OpenCV format
            before_pil = self.before_pdf_pages[before_page_idx]
            after_pil = self.after_pdf_pages[after_page_idx]
            
            before_img = cv2.cvtColor(np.array(before_pil), cv2.COLOR_RGB2BGR)
            after_img = cv2.cvtColor(np.array(after_pil), cv2.COLOR_RGB2BGR)
        else:
            # For regular images, load from file paths
            before_img = cv2.imread(self.before_image_path)
            after_img = cv2.imread(self.after_image_path)

        if before_img is None or after_img is None:
            QMessageBox.warning(self, "Error", "Could not read one or both images.")
            return

        # 1) Try ORB alignment
        aligned_after = self.align_images_orb(before_img, after_img)
        # 2) If ORB fails AND SIFT is enabled, try SIFT
        if aligned_after is None and self.sift_enabled:
            aligned_after = self.align_images_sift(before_img, after_img)

        if aligned_after is None:
            QMessageBox.warning(self, "Alignment Failed", 
                                "Could not align the images (not enough good matches).")
            return

        # 3) Enhanced difference detection
        diff_result = self.get_difference_enhanced(before_img, aligned_after)
        self.show_diff_image(diff_result)

    # ------------------- Alignment Methods (ORB & SIFT) -------------------
    def align_images_orb(self, before_img, after_img):
        """Use ORB feature detection/matching to warp after_img to match before_img."""
        return self.align_images_generic(before_img, after_img, method="ORB")

    def align_images_sift(self, before_img, after_img):
        """Use SIFT feature detection/matching to warp after_img to match before_img."""
        return self.align_images_generic(before_img, after_img, method="SIFT")

    def align_images_generic(self, before_img, after_img, method="ORB"):
        """
        A generic alignment method that uses either ORB or SIFT,
        depending on 'method' parameter. Returns the aligned 'after' image
        or None if alignment fails.
        """
        before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

        if method == "SIFT":
            # Requires opencv-contrib-python
            # noinspection PyUnresolvedReferences
            detector = cv2.xfeatures2d.SIFT_create()
            norm_type = cv2.NORM_L2
        else:
            # default to ORB
            detector = cv2.ORB_create(5000)
            norm_type = cv2.NORM_HAMMING

        kp1, des1 = detector.detectAndCompute(before_gray, None)
        kp2, des2 = detector.detectAndCompute(after_gray, None)

        if des1 is None or des2 is None:
            return None

        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        # KNN match
        matches_knn = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        ratio_thresh = 0.7
        for m, n in matches_knn:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        MIN_MATCH_COUNT = 10
        if len(good_matches) < MIN_MATCH_COUNT:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is None:
            return None

        h, w, _ = before_img.shape
        aligned_after = cv2.warpPerspective(after_img, M, (w, h))
        return aligned_after

    # ------------------- Enhanced Difference in Lab -------------------
    def get_difference_enhanced(self, before_img, aligned_after):
        """
        Convert both images to Lab color space, blur them,
        compute the absolute difference, threshold, morphological
        filtering, and draw bounding boxes around changes.
        Also calculates a difference score.
        """
        # Convert to Lab color space
        before_lab = cv2.cvtColor(before_img, cv2.COLOR_BGR2Lab)
        after_lab  = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2Lab)

        # Optional: blur to reduce noise
        before_blur = cv2.GaussianBlur(before_lab, (5, 5), 0)
        after_blur  = cv2.GaussianBlur(after_lab,  (5, 5), 0)

        # Compute absolute difference (3-channel)
        diff = cv2.absdiff(before_blur, after_blur)
        # Convert to a single channel by taking max across Lab channels
        diff_gray = np.max(diff, axis=2).astype(np.uint8)

        # Retrieve user-chosen threshold from slider
        user_thresh = self.threshold_slider.value()
        _, diff_thresh = cv2.threshold(diff_gray, user_thresh, 255, cv2.THRESH_BINARY)

        # Morphological operations
        kernel_size = self.kernel_slider.value()  # user-chosen kernel size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # "Close" to fill holes
        diff_morph = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        # (Optionally "Open" afterwards if desired)
        # diff_morph = cv2.morphologyEx(diff_morph, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(diff_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result_img = aligned_after.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # skip very small areas
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Calculate & display difference score (% of changed pixels)
        nonzero_pixels = cv2.countNonZero(diff_morph)
        total_pixels = diff_morph.shape[0] * diff_morph.shape[1]
        percentage_diff = (nonzero_pixels / total_pixels) * 100
        self.diff_score_label.setText(f"Difference Score: {percentage_diff:.2f}%")

        # Store for saving
        self.latest_diff_image = result_img.copy()

        return result_img

    # ------------------- Display & Save Results -------------------
    def show_diff_image(self, diff_img_cv):
        """Convert OpenCV BGR image to QImage and display in diff_label with zoom support."""
        diff_rgb = cv2.cvtColor(diff_img_cv, cv2.COLOR_BGR2RGB)
        height, width, channel = diff_rgb.shape
        bytes_per_line = channel * width
        qimg = QImage(diff_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Apply zoom scale
        scale = self.zoom_scales["diff"]
        scaled_size = pixmap.size() * scale
        self.diff_label.setPixmap(pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.diff_label.setFixedSize(scaled_size)

    def save_diff_image(self):
        """Save the last difference image (with bounding boxes) to disk."""
        if self.latest_diff_image is None:
            QMessageBox.warning(self, "No Difference Image", "Please compare images first.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save Difference Image",
            "difference_result.jpg",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)",
            options=options,
        )
        if file_name:
            cv2.imwrite(file_name, self.latest_diff_image)
            QMessageBox.information(self, "Saved", f"File saved as {file_name}")

    # ------------------- Utility: Pillow→QImage -------------------
    @staticmethod
    def pil_to_qimage(img):
        """Convert a Pillow Image to QImage."""
        img = img.convert("RGBA")
        data = img.tobytes("raw", "RGBA")
        q_img = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
        return q_img
        
    @staticmethod
    def qimage_to_pil(qimage):
        """Convert QImage to PIL Image"""
        buffer = QImage(qimage)
        size = buffer.size()
        
        # Convert to correct format for PIL
        buffer = buffer.convertToFormat(QImage.Format_RGBA8888)
        
        # Extract data
        ptr = buffer.bits()
        ptr.setsize(buffer.byteCount())
        arr = np.array(ptr).reshape(size.height(), size.width(), 4)
        
        # Create PIL Image
        return Image.fromarray(arr, 'RGBA').convert('RGB')

    # ------------------- PDF Processing Methods -------------------
    def pdf_to_pil_image(self, pdf_page, dpi=200):
        """Convert a PyPDF2 page to a PIL Image using a more reliable method"""
        # Calculate dimensions based on mediabox and DPI
        width = int(float(pdf_page.mediabox.width) * dpi/72)
        height = int(float(pdf_page.mediabox.height) * dpi/72)
        
        # Create a writer for this single page
        writer = PyPDF2.PdfWriter()
        writer.add_page(pdf_page)
        
        # Write to a bytes buffer
        pdf_bytes = io.BytesIO()
        writer.write(pdf_bytes)
        pdf_bytes.seek(0)
        
        # Create a blank RGB image as fallback
        blank_img = Image.new('RGB', (width, height), (255, 255, 255))
        
        try:
            # Try to convert PDF to image using PIL
            temp_img = Image.open(pdf_bytes)
            if temp_img.mode == 'RGBA':
                temp_img = temp_img.convert('RGB')
            return temp_img
        except Exception as e:
            # First attempt failed, try alternative method
            try:
                # Alternative method: render PDF using PyPDF2's extractText and create an image with text
                text = pdf_page.extract_text()
                
                # If we got text, create a simple text rendering
                if text and text.strip():
                    from PIL import ImageDraw, ImageFont
                    img = blank_img.copy()
                    draw = ImageDraw.Draw(img)
                    
                    # Use a default font
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except IOError:
                        font = ImageFont.load_default()
                    
                    # Draw the text
                    draw.text((10, 10), text, fill=(0, 0, 0), font=font)
                    return img
                else:
                    # No text extracted, return blank image
                    return blank_img
            except Exception:
                # If all methods fail, return the blank image
                return blank_img
    
    def process_pdf_file(self, pdf_path, which):
        """Process a PDF file, extract pages as images, and display the first page"""
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                if page_count == 0:
                    raise ValueError("The PDF file has no pages")
                
                # Update page count and spinner
                if which == "before":
                    self.before_pdf_page_count = page_count
                    self.before_page_spinner.setMaximum(page_count)
                    self.before_page_spinner.setValue(1)  # Start with first page
                    self.before_page_spinner.setEnabled(True)
                    self.before_pdf_pages = []
                else:  # after
                    self.after_pdf_page_count = page_count
                    self.after_page_spinner.setMaximum(page_count)
                    self.after_page_spinner.setValue(1)  # Start with first page
                    self.after_page_spinner.setEnabled(True)
                    self.after_pdf_pages = []
                
                # Convert PDF pages to images
                temp_dir = self.before_temp_dir if which == "before" else self.after_temp_dir
                images = []
                
                # Process each page
                for page_num in range(page_count):
                    # Get the page
                    page = pdf_reader.pages[page_num]
                    
                    # Convert page to PIL Image
                    img = self.pdf_to_pil_image(page, dpi=200)
                    
                    # Save the image to the temp directory
                    img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")
                    img.save(img_path)
                    
                    # Add to our list of images
                    images.append(img)
                
                # Store the images
                if which == "before":
                    self.before_pdf_pages = images
                    # Display the first page
                    self.show_pdf_page(0, self.before_pdf_pages, self.before_label)
                else:  # after
                    self.after_pdf_pages = images
                    # Display the first page
                    self.show_pdf_page(0, self.after_pdf_pages, self.after_label)
                
        except Exception as e:
            QMessageBox.warning(self, "PDF Processing Error", str(e))
            # Clean up on error
            if which == "before":
                self.before_is_pdf = False
                self.before_image_path = None
            else:  # after
                self.after_is_pdf = False
                self.after_image_path = None
    
    def show_pdf_page(self, page_index, pdf_pages, label_widget):
        """Display a specific page from the PDF in the given label with zoom support"""
        if 0 <= page_index < len(pdf_pages):
            img = pdf_pages[page_index]
            qimg = self.pil_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)
            
            # Get the label type and scale
            label_type = "before" if label_widget == self.before_label else "after" if label_widget == self.after_label else "diff"
            scale = self.zoom_scales[label_type]
            
            # Scale the image
            scaled_size = pixmap.size() * scale
            label_widget.setPixmap(pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            label_widget.setFixedSize(scaled_size)
    
    def clear_before_image(self):
        """Clear the 'before' image/PDF and reset its label"""
        self.before_image_path = None
        self.before_label.setText("Before Image")
        self.before_label.setPixmap(QPixmap())
        self.latest_diff_image = None
        self.diff_label.setText("Difference Highlight")
        self.diff_label.setPixmap(QPixmap())
        self.diff_score_label.setText("Difference Score: N/A")
        
        # Clean up PDF resources if needed
        if self.before_is_pdf:
            self.before_is_pdf = False
            self.before_pdf_pages = []
            self.before_pdf_page_count = 0
            self.before_page_spinner.setValue(1)
            self.before_page_spinner.setEnabled(False)
            
            # Clean up temp directory
            if self.before_temp_dir and os.path.exists(self.before_temp_dir):
                for file in os.listdir(self.before_temp_dir):
                    os.remove(os.path.join(self.before_temp_dir, file))
                os.rmdir(self.before_temp_dir)
                self.before_temp_dir = None

    def clear_after_image(self):
        """Clear the 'after' image/PDF and reset its label"""
        self.after_image_path = None
        self.after_label.setText("After Image")
        self.after_label.setPixmap(QPixmap())
        self.latest_diff_image = None
        self.diff_label.setText("Difference Highlight")
        self.diff_label.setPixmap(QPixmap())
        self.diff_score_label.setText("Difference Score: N/A")
        
        # Clean up PDF resources if needed
        if self.after_is_pdf:
            self.after_is_pdf = False
            self.after_pdf_pages = []
            self.after_pdf_page_count = 0
            self.after_page_spinner.setValue(1)
            self.after_page_spinner.setEnabled(False)
            
            # Clean up temp directory
            if self.after_temp_dir and os.path.exists(self.after_temp_dir):
                for file in os.listdir(self.after_temp_dir):
                    os.remove(os.path.join(self.after_temp_dir, file))
                os.rmdir(self.after_temp_dir)
                self.after_temp_dir = None

    def clear_all(self):
        """Clear all images/PDFs and reset to initial state"""
        self.clear_before_image()
        self.clear_after_image()
        # Reset sliders to default values
        self.threshold_slider.setValue(50)
        self.kernel_slider.setValue(5)
        
    def __del__(self):
        """Destructor to clean up temporary directories when the application closes"""
        # Clean up before temp directory
        if hasattr(self, 'before_temp_dir') and self.before_temp_dir and os.path.exists(self.before_temp_dir):
            try:
                for file in os.listdir(self.before_temp_dir):
                    os.remove(os.path.join(self.before_temp_dir, file))
                os.rmdir(self.before_temp_dir)
            except Exception:
                pass
                
        # Clean up after temp directory
        if hasattr(self, 'after_temp_dir') and self.after_temp_dir and os.path.exists(self.after_temp_dir):
            try:
                for file in os.listdir(self.after_temp_dir):
                    os.remove(os.path.join(self.after_temp_dir, file))
                os.rmdir(self.after_temp_dir)
            except Exception:
                pass
        # Removed UI element access that was causing exceptions on application close

def main():
    app = QApplication(sys.argv)
    window = ImageDiffApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
