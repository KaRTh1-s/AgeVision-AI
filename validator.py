"""
ImageValidator — Pure OpenCV face detection (no external dependencies).
Replaces MTCNN with Haar Cascade + DNN-based face detector.
Works 100% offline with just opencv-python-headless.
"""
import cv2
import numpy as np


HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_cascade = None

def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(HAAR_PATH)
    return _cascade


class ImageValidator:
    def __init__(self, device=None):
        # device param kept for API compatibility — not used by OpenCV
        self.min_face_size   = 80    # pixels (width & height)
        self.min_blur_var    = 100   # Laplacian variance threshold
        self.min_scale_ratio = 0.05  # face must be ≥ 5% of image width

    # ── helpers ────────────────────────────────────────────────────────────
    def _blur_score(self, gray_crop):
        return cv2.Laplacian(gray_crop, cv2.CV_64F).var()

    def _detect_faces(self, image_bgr):
        """Returns list of (x, y, w, h) rectangles, largest first."""
        cascade = _get_cascade()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return []
        # sort largest area first
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return faces

    # ── public API ─────────────────────────────────────────────────────────
    def validate(self, image_bgr):
        """
        Returns (is_valid: bool, message: str, box: [x1,y1,x2,y2] | None)
        """
        h_img, w_img = image_bgr.shape[:2]

        # 1. Minimum image size sanity check
        if w_img < 64 or h_img < 64:
            return False, "Image resolution is too low. Please upload a larger photo.", None

        # 2. Detect faces
        faces = self._detect_faces(image_bgr)
        if not faces:
            return False, "No human face detected. Please upload a clear portrait photo.", None

        # Pick the largest face
        x, y, w, h = faces[0]

        # 3. Face too small relative to image
        if w < self.min_face_size or h < self.min_face_size:
            return False, (
                f"Face is too small ({w}×{h} px). "
                "Please upload a closer portrait photo."
            ), None

        if w / w_img < self.min_scale_ratio:
            return False, "Face is too far from the camera. Please move closer.", None

        # 4. Blur check on face crop
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        face_crop = image_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return False, "Could not crop face region. Please try a different photo.", None

        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        blur = self._blur_score(gray_face)
        if blur < self.min_blur_var:
            return False, (
                f"Image is too blurry (sharpness score: {blur:.0f}). "
                "Please upload a sharper photo."
            ), None

        return True, "Valid", [x1, y1, x2, y2]
