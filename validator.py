"""
ImageValidator — Robust face validation for both local and cloud environments.
When cv2.CascadeClassifier is unavailable (some headless builds), falls back to
multi-signal heuristics: skin-tone ratio, edge density, white-background ratio,
and unique color count to reject cartoons, memes, sketches, and animals.
"""
import os
import cv2
import numpy as np


# ── Cascade loader with auto-download fallback ─────────────────────────────
def _load_cascade():
    """Try local, then download from GitHub. Returns None if CascadeClassifier unavailable."""
    
    # Quick check: does this build even have CascadeClassifier?
    if not hasattr(cv2, "CascadeClassifier"):
        print("[WARN] cv2.CascadeClassifier not in this build — using heuristic validation.")
        return None

    # 1. Built-in cv2 data path
    try:
        p = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(p) and os.path.getsize(p) > 10000:
            c = cv2.CascadeClassifier(p)
            if not c.empty():
                print("[OK] Haar cascade loaded from cv2.data")
                return c
    except Exception:
        pass

    # 2. Cached local copy
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    if os.path.exists(local) and os.path.getsize(local) > 10000:
        try:
            c = cv2.CascadeClassifier(local)
            if not c.empty():
                print("[OK] Haar cascade loaded from local cache")
                return c
        except Exception:
            pass

    # 3. Download from OpenCV GitHub
    url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
    try:
        import urllib.request
        print("[INFO] Downloading Haar cascade from GitHub…")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        with open(local, "wb") as f:
            f.write(data)
        c = cv2.CascadeClassifier(local)
        if not c.empty():
            print("[OK] Haar cascade downloaded and loaded")
            return c
    except Exception as e:
        print(f"[WARN] Cascade download failed: {e}")

    print("[WARN] Haar cascade unavailable — heuristic validation active")
    return None


_CASCADE = _load_cascade()


class ImageValidator:
    def __init__(self, device=None):
        self.min_face_px  = 60
        self.min_blur_var = 50

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _blur_score(gray):
        try:
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 9999.0

    @staticmethod
    def _skin_ratio(image_bgr):
        """Fraction of pixels in human skin-tone HSV range."""
        try:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
            mask = (
                ((h < 25) | (h > 160)) &   # hue: skin-tone
                (s > 18) & (s < 175) &      # not grey, not oversaturated
                (v > 60)                    # not too dark
            )
            return float(np.mean(mask))
        except Exception:
            return 0.1  # assume OK if we can't compute

    @staticmethod
    def _edge_ratio(gray):
        """Fraction of edge pixels (cartoons have very sharp lines)."""
        try:
            edges = cv2.Canny(gray, 50, 150)
            return float(np.mean(edges > 0))
        except Exception:
            return 0.05

    @staticmethod
    def _white_ratio(gray):
        """Fraction of near-white pixels (memes often have white backgrounds)."""
        try:
            return float(np.mean(gray > 228))
        except Exception:
            return 0.0

    @staticmethod
    def _unique_color_count(image_bgr):
        """Count distinct colors in a 50×50 thumbnail (cartoons have fewer)."""
        try:
            small = cv2.resize(image_bgr, (50, 50))
            return len(np.unique(small.reshape(-1, 3), axis=0))
        except Exception:
            return 500

    # ── heuristic gate (used when Haar cascade is unavailable) ─────────────
    def _heuristic_validate(self, image_bgr):
        """
        Returns (is_valid: bool, message: str)
        Rejects cartoons, memes, sketches, and non-portrait images.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        skin  = self._skin_ratio(image_bgr)
        edges = self._edge_ratio(gray)
        white = self._white_ratio(gray)
        colors= self._unique_color_count(image_bgr)

        # ── Decision rules ─────────────────────────────────────────────────
        # 1. Cartoon / sketch: sharp edges + almost no skin tone
        if edges > 0.12 and skin < 0.04:
            return False, "Image appears to be a cartoon, sketch, or illustration. Please upload a real portrait photo."

        # 2. Meme with white background + little/no skin
        if white > 0.40 and skin < 0.06:
            return False, "No human face detected. Please upload a clear portrait photo of a person."

        # 3. Very simple graphic (few colors) + no skin
        if colors < 120 and skin < 0.04:
            return False, "Image appears to be a simple graphic or illustration, not a portrait photo."

        # 4. Animal / object photo with zero skin tone
        if skin < 0.015:
            return False, "No human face detected. Please upload a clear portrait photo of a person."

        # 5. Blur check on whole image
        blur = self._blur_score(gray)
        if blur < self.min_blur_var:
            return False, f"Image is too blurry (sharpness: {blur:.0f}). Please upload a sharper photo."

        return True, "Valid"

    # ── Haar cascade detection ─────────────────────────────────────────────
    def _detect_faces_cascade(self, image_bgr):
        """Returns face list (largest first), or None if cascade unavailable."""
        if _CASCADE is None:
            return None
        try:
            gray = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
            for sf, mn in [(1.1, 4), (1.15, 3), (1.2, 3)]:
                faces = _CASCADE.detectMultiScale(
                    gray, scaleFactor=sf, minNeighbors=mn,
                    minSize=(self.min_face_px, self.min_face_px),
                )
                if len(faces) > 0:
                    return sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            return []
        except Exception:
            return None

    # ── Public API ─────────────────────────────────────────────────────────
    def validate(self, image_bgr):
        """
        Returns (is_valid: bool, message: str, box: [x1,y1,x2,y2] | None)
        """
        if image_bgr is None or image_bgr.size == 0:
            return False, "Empty or corrupt image. Please upload a valid photo.", None

        h_img, w_img = image_bgr.shape[:2]
        if w_img < 64 or h_img < 64:
            return False, "Image resolution is too low. Please upload a larger photo.", None

        faces = self._detect_faces_cascade(image_bgr)

        if faces is not None:
            # ── Haar cascade path ──────────────────────────────────────────
            if len(faces) == 0:
                return False, "No human face detected. Please upload a clear portrait photo.", None

            x, y, w, h = faces[0]
            if w < self.min_face_px or h < self.min_face_px:
                return False, f"Face is too small ({w}×{h}px). Please upload a closer portrait.", None

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x+w), min(h_img, y+h)
            face_crop = image_bgr[y1:y2, x1:x2]

            if face_crop.size > 0:
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur = self._blur_score(gray_face)
                if blur < self.min_blur_var:
                    return False, f"Image is too blurry (sharpness: {blur:.0f}). Please upload a sharper photo.", None

            return True, "Valid", [x1, y1, x2, y2]

        else:
            # ── Heuristic path (cascade unavailable) ──────────────────────
            ok, msg = self._heuristic_validate(image_bgr)
            if not ok:
                return False, msg, None
            return True, "Valid", [0, 0, w_img, h_img]
