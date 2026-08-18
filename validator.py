"""
ImageValidator — Robust face validation with graceful fallback.
Uses OpenCV Haar Cascade when available; falls back to blur-only check on cloud.
"""
import cv2
import numpy as np


def _load_cascade():
    """Load Haar cascade; return None if unavailable (e.g., stripped headless builds)."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            return None
        return cascade
    except (AttributeError, Exception):
        return None


_CASCADE = _load_cascade()


class ImageValidator:
    def __init__(self, device=None):
        self.min_face_size   = 80
        self.min_blur_var    = 60   # slightly relaxed for cloud
        self.min_scale_ratio = 0.04

    def _blur_score(self, gray):
        try:
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 9999.0   # assume sharp if we can't compute

    def _detect_faces(self, image_bgr):
        """Returns sorted list of (x,y,w,h), largest first. Empty list = no face."""
        if _CASCADE is None:
            return None   # None = cascade unavailable (don't reject)
        try:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = _CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(self.min_face_size, self.min_face_size),
            )
            if len(faces) == 0:
                return []
            return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        except Exception:
            return None   # treat as cascade unavailable

    def validate(self, image_bgr):
        """
        Returns (is_valid: bool, message: str, box: [x1,y1,x2,y2] | None)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, "Empty or corrupt image. Please upload a valid photo.", None

        h_img, w_img = image_bgr.shape[:2]

        # 1. Basic size sanity
        if w_img < 64 or h_img < 64:
            return False, "Image resolution is too low. Please upload a larger photo.", None

        # 2. Face detection (optional — skip if cascade unavailable)
        faces = self._detect_faces(image_bgr)
        if faces is not None and len(faces) == 0:
            return False, "No human face detected. Please upload a clear portrait photo.", None

        # 3. Blur check (whole image if no crop available)
        try:
            gray_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(w_img, x + w), min(h_img, y + h)
                face_crop = image_bgr[y1:y2, x1:x2]
                gray_check = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.size > 0 else gray_img
            else:
                gray_check = gray_img
                x1, y1, x2, y2 = 0, 0, w_img, h_img

            blur = self._blur_score(gray_check)
            if blur < self.min_blur_var:
                return False, (
                    f"Image is too blurry (sharpness: {blur:.0f}). "
                    "Please upload a sharper photo."
                ), None
        except Exception:
            x1, y1, x2, y2 = 0, 0, w_img, h_img

        return True, "Valid", [x1, y1, x2, y2]
