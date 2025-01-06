import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QFileDialog, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image

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

        self.setWindowTitle("Image Difference Viewer (Full Enhanced)")
        self.setGeometry(200, 200, 1300, 700)

        # Main container widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Overall layout
        self.layout = QVBoxLayout()
        
        # Top: Buttons + parameter sliders
        self.top_layout = QHBoxLayout()
        
        # Middle: Three image labels
        self.images_layout = QHBoxLayout()
        
        # Bottom: Extra info (score) & save button
        self.bottom_layout = QHBoxLayout()

        # === Buttons for loading/comparing images ===
        self.load_before_btn = QPushButton("Load 'Before' Image")
        self.load_after_btn = QPushButton("Load 'After' Image")
        self.compare_btn = QPushButton("Compare Images")
        self.save_result_btn = QPushButton("Save Result")

        self.load_before_btn.clicked.connect(self.load_before_image)
        self.load_after_btn.clicked.connect(self.load_after_image)
        self.compare_btn.clicked.connect(self.compare_images)
        self.save_result_btn.clicked.connect(self.save_diff_image)

        # Add buttons to top_layout
        self.top_layout.addWidget(self.load_before_btn)
        self.top_layout.addWidget(self.load_after_btn)
        self.top_layout.addWidget(self.compare_btn)
        self.top_layout.addWidget(self.save_result_btn)

        # === Sliders for threshold + kernel size ===
        self.threshold_label = QLabel("Threshold: 30")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        self.kernel_label = QLabel("Kernel Size: 5")
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(1, 15)
        self.kernel_slider.setValue(5)
        self.kernel_slider.valueChanged.connect(self.update_kernel_label)

        # Add sliders/labels to top_layout
        self.top_layout.addWidget(self.threshold_label)
        self.top_layout.addWidget(self.threshold_slider)
        self.top_layout.addWidget(self.kernel_label)
        self.top_layout.addWidget(self.kernel_slider)

        # === Image Labels ===
        self.before_label = QLabel("Before Image")
        self.before_label.setAlignment(Qt.AlignCenter)
        self.before_label.setScaledContents(True)
        self.before_label.setMinimumSize(300, 300)

        self.after_label = QLabel("After Image")
        self.after_label.setAlignment(Qt.AlignCenter)
        self.after_label.setScaledContents(True)
        self.after_label.setMinimumSize(300, 300)

        self.diff_label = QLabel("Difference Highlight")
        self.diff_label.setAlignment(Qt.AlignCenter)
        self.diff_label.setScaledContents(True)
        self.diff_label.setMinimumSize(300, 300)

        self.images_layout.addWidget(self.before_label)
        self.images_layout.addWidget(self.after_label)
        self.images_layout.addWidget(self.diff_label)

        # === Bottom layout: difference score display ===
        self.diff_score_label = QLabel("Difference Score: N/A")
        self.bottom_layout.addWidget(self.diff_score_label)

        # Combine layouts
        self.layout.addLayout(self.top_layout)
        self.layout.addLayout(self.images_layout)
        self.layout.addLayout(self.bottom_layout)
        self.main_widget.setLayout(self.layout)

        # Store file paths
        self.before_image_path = None
        self.after_image_path = None

        # Keep track of the latest difference image for saving
        self.latest_diff_image = None

        # Check if SIFT is available
        self.sift_enabled = is_sift_available()

    # ------------------- UI Update Methods -------------------
    def update_threshold_label(self, value):
        self.threshold_label.setText(f"Threshold: {value}")

    def update_kernel_label(self, value):
        self.kernel_label.setText(f"Kernel Size: {value}")

    # ------------------- Load Images -------------------
    def load_before_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Before Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)",
            options=options,
        )
        if file_name:
            self.before_image_path = file_name
            self.show_image_in_label(file_name, self.before_label)

    def load_after_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select After Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)",
            options=options,
        )
        if file_name:
            self.after_image_path = file_name
            self.show_image_in_label(file_name, self.after_label)

    def show_image_in_label(self, image_path, label_widget):
        """
        Helper method to load an image using Pillow,
        convert to QPixmap, and display in the given QLabel.
        """
        try:
            img = Image.open(image_path)
            qimg = self.pil_to_qimage(img)
            pixmap = QPixmap.fromImage(qimg)
            label_widget.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.warning(self, "Image Load Error", str(e))

    # ------------------- Compare Images -------------------
    def compare_images(self):
        """
        Align the 'After' image to the 'Before' image (ORB → fallback to SIFT),
        then do an enhanced difference detection in Lab color space,
        using the user-chosen threshold and kernel size.
        """
        if not self.before_image_path or not self.after_image_path:
            QMessageBox.warning(self, "Missing Images", "Please load both 'Before' and 'After' images first.")
            return

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
        """Convert OpenCV BGR image to QImage and display in diff_label."""
        diff_rgb = cv2.cvtColor(diff_img_cv, cv2.COLOR_BGR2RGB)
        height, width, channel = diff_rgb.shape
        bytes_per_line = channel * width
        qimg = QImage(diff_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.diff_label.setPixmap(pixmap)

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

def main():
    app = QApplication(sys.argv)
    window = ImageDiffApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
