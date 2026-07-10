"""J.A.R.V.I.S. Spatial Workspace - camera-first, gesture-controlled HUD.

Controls
--------
* Clap (visually) to dismiss the welcome screen.
* SELECTION IS DWELL, NOT PINCH: point at a target and hold steady; a ring
  fills, then it selects. This is used for menu items, drives, folders, files
  and every on-screen button - one clear, unambiguous gesture everywhere.
* Explore Device -> dwell a drive -> dwell a folder -> dwell a file, i.e.
  drive -> folder -> file, in that order. Files open in-app at full resolution.
* SCANNER: hold a page up, thumbs-up to capture it, dwell SAVE PDF to write a
  multi-page scan into ~/JARVIS_Scans (it then shows up in the device browser).
* Open BOTH palms over the archive to drag it; spread / squeeze both palms to
  resize the archive.
* Two-hand PINCH spread / squeeze zooms the ENTIRE HUD (keys + / - / 0 too),
  giving more workspace without losing resolution.
* Hold a closed fist to go back one level.  m: main menu.  c: cycle studio
  filter.  f: fullscreen.  q / ESC: quit.
"""
import cv2
import math
import os
import platform
import time
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import mediapipe as mp
import numpy as np

WIN = "J.A.R.V.I.S. Spatial Workspace"
FONT = cv2.FONT_HERSHEY_SIMPLEX
TECH_BLUE = (255, 220, 50)
TECH_BLUE_DARK = (180, 100, 0)
TECH_WHITE = (255, 250, 240)
TECH_ACCENT = (255, 180, 30)
TECH_GREEN = (120, 255, 120)
HUD_BG = (35, 20, 5)
DETECT_WIDTH = 640                 # deliberately lower-quality computer feed
WELCOME_TEXT = "WELCOME BACK, HRISHI"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_MAX_DIM = 1600                # full-fidelity cap for in-app document view
SCAN_DIR = Path.home() / "JARVIS_Scans"
STUDIO_PRESETS = ["STUDIO", "CINEMA", "CLARITY", "NOIR"]


def glow_text(img, text, pos, scale=0.6, color=TECH_BLUE, thick=1, centered=False):
    if centered:
        width = cv2.getTextSize(text, FONT, scale, thick)[0][0]
        pos = (int(pos[0] - width / 2), pos[1])
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def wrap_label(text, max_chars):
    """Break a filename onto multiple lines so it stays fully readable."""
    raw, words = text.replace("_", " ").split(), []
    for word in raw:
        while len(word) > max_chars:
            words.append(word[:max_chars]); word = word[max_chars:]
        if word:
            words.append(word)
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3] or [text[:max_chars]]


def translucent_rect(img, p1, p2, color, alpha):
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(img.shape[1], int(p2[0])), min(img.shape[0], int(p2[1]))
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def corner_brackets(img, margin=20, length=32, color=TECH_BLUE, thick=2):
    h, w = img.shape[:2]
    for p, a, b in [((margin, margin), (margin + length, margin), (margin, margin + length)),
                    ((w-margin, margin), (w-margin-length, margin), (w-margin, margin + length)),
                    ((margin, h-margin), (margin+length, h-margin), (margin, h-margin-length)),
                    ((w-margin, h-margin), (w-margin-length, h-margin), (w-margin, h-margin-length))]:
        cv2.line(img, p, a, color, thick, cv2.LINE_AA)
        cv2.line(img, p, b, color, thick, cv2.LINE_AA)


def reticle(img, point, t):
    if point is None:
        return
    x, y = map(int, point)
    r = int(18 + 3 * math.sin(t * 4))
    for k in range(4):
        angle = math.radians((t * 90) + k * 90)
        p1 = (int(x + math.cos(angle)*(r+4)), int(y + math.sin(angle)*(r+4)))
        p2 = (int(x + math.cos(angle)*(r+12)), int(y + math.sin(angle)*(r+12)))
        cv2.line(img, p1, p2, TECH_BLUE, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, TECH_BLUE, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, TECH_WHITE, -1, cv2.LINE_AA)


def draw_dwell_ring(img, point, progress):
    """A filling arc around the pointer that shows dwell-to-select progress."""
    if point is None or progress <= 0.01:
        return
    x, y = map(int, point)
    r = 30
    cv2.circle(img, (x, y), r, TECH_BLUE_DARK, 2, cv2.LINE_AA)
    cv2.ellipse(img, (x, y), (r, r), -90, 0, int(360 * min(1.0, progress)), TECH_ACCENT, 4, cv2.LINE_AA)
    if progress >= 1.0:
        cv2.circle(img, (x, y), r + 6, TECH_ACCENT, 2, cv2.LINE_AA)


class StudioCameraFilter:
    """Edge-preserving temporal denoise, then a selectable studio colour grade."""
    def __init__(self):
        self.previous_clean, self.previous_raw = None, None
        self.preset, self.vignette, self.vignette_shape = 0, None, None
    def cycle(self):
        self.preset = (self.preset + 1) % len(STUDIO_PRESETS)
    def name(self):
        return STUDIO_PRESETS[self.preset]

    def _vignette_mask(self, shape):
        if self.vignette_shape == shape:
            return self.vignette
        h, w = shape[:2]
        kernel = cv2.getGaussianKernel(h, h * 0.6) @ cv2.getGaussianKernel(w, w * 0.6).T
        kernel = kernel / kernel.max()
        self.vignette = (0.6 + 0.4 * kernel).astype(np.float32)[..., None]
        self.vignette_shape = shape
        return self.vignette

    def _grade(self, img):
        name = STUDIO_PRESETS[self.preset]
        if name == "NOIR":
            gray = cv2.createCLAHE(2.6, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img = cv2.convertScaleAbs(img, alpha=1.08, beta=-4)
        blue, green, red = cv2.split(img)
        if name == "STUDIO":
            img = cv2.merge((cv2.add(blue, 6), green, cv2.subtract(red, 2)))
        elif name == "CINEMA":                     # cool shadows, warm highlights
            img = cv2.convertScaleAbs(cv2.merge((cv2.add(blue, 10), green, cv2.add(red, 7))), alpha=1.05)
        elif name == "CLARITY":
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            luma, a, b = cv2.split(lab)
            luma = cv2.createCLAHE(2.0, (8, 8)).apply(luma)
            img = cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)
        return img

    def apply(self, frame):
        # 1) Edge-preserving spatial denoise, blended temporally on static pixels
        #    so a still scene converges to a clean, noise-free plate.
        spatial = cv2.bilateralFilter(frame, 9, 34, 14)
        if self.previous_clean is not None and self.previous_clean.shape == spatial.shape:
            difference = cv2.cvtColor(cv2.absdiff(frame, self.previous_raw), cv2.COLOR_BGR2GRAY)
            moving = cv2.GaussianBlur((difference > 14).astype(np.uint8) * 255, (0, 0), 2.4)
            stable = cv2.addWeighted(spatial, .5, self.previous_clean, .5, 0)
            mask = cv2.cvtColor(moving, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255
            clean = (spatial * mask + stable * (1 - mask)).astype(np.uint8)
        else:
            clean = spatial
        self.previous_clean, self.previous_raw = clean, frame.copy()
        # 2) Studio grade, then clarity (unsharp) and a soft vignette.
        graded = self._grade(clean)
        soft = cv2.GaussianBlur(graded, (0, 0), 1.1)
        graded = cv2.addWeighted(graded, 1.22, soft, -.22, 0)
        mask = self._vignette_mask(graded.shape)
        return np.clip(graded.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def low_quality_tracking_frame(frame):
    h, w = frame.shape[:2]
    if w <= DETECT_WIDTH:
        return frame
    return cv2.resize(frame, (DETECT_WIDTH, max(1, int(h * DETECT_WIDTH / w))), interpolation=cv2.INTER_AREA)


def apply_hud_zoom(img, zoom):
    """Digital zoom of the whole composited HUD. >1 crops in (keeps detail);
    <1 shrinks the frame onto a dark canvas, freeing extra workspace."""
    if abs(zoom - 1.0) < 0.02:
        return img
    h, w = img.shape[:2]
    if zoom > 1.0:
        cw, ch = int(w / zoom), int(h / zoom)
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        return cv2.resize(img[y0:y0+ch, x0:x0+cw], (w, h), interpolation=cv2.INTER_LINEAR)
    nw, nh = max(1, int(w * zoom)), max(1, int(h * zoom))
    canvas = np.zeros_like(img)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    corner_brackets(canvas)
    return canvas


def paste_texture(img, texture, points, brightness=1.0):
    dst = np.float32(points)
    th, tw = texture.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), dst)
    source = cv2.convertScaleAbs(texture, alpha=brightness) if brightness != 1 else texture
    warped = cv2.warpPerspective(source, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst), (255, 255, 255))
    return cv2.add(cv2.bitwise_and(img, cv2.bitwise_not(mask)), cv2.bitwise_and(warped, mask)), np.int32(dst)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def create_file_preview(file_path):
    path = Path(file_path)
    if path.is_dir(): return create_directory_card(path)
    suffix = path.suffix.lower().lstrip(".") or "FILE"
    if path.suffix.lower() in IMAGE_SUFFIXES:
        photo = cv2.imread(str(path))
        if photo is not None:
            card = cv2.resize(photo, (400, 560), interpolation=cv2.INTER_AREA)
            cv2.rectangle(card, (0, 500), (400, 560), (12, 26, 38), -1)
            cv2.putText(card, path.name[:30], (16, 538), FONT, .6, TECH_WHITE, 1, cv2.LINE_AA)
            return card
    img = np.full((560, 400, 3), (22, 30, 38), np.uint8)
    cv2.rectangle(img, (0, 0), (400, 82), (45, 84, 110), -1)
    cv2.putText(img, "DEVICE FILE", (20, 52), FONT, .8, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, suffix[:8].upper(), (24, 190), FONT, 1.7, TECH_ACCENT, 3, cv2.LINE_AA)
    for row, line in enumerate(wrap_label(path.name, 22)):
        cv2.putText(img, line, (24, 262 + row * 40), FONT, .78, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "HOLD POINT TO SELECT", (24, 500), FONT, .58, TECH_BLUE, 1, cv2.LINE_AA)
    cv2.putText(img, "THUMBS-UP CONFIRMS OPEN", (24, 532), FONT, .5, TECH_BLUE_DARK, 1, cv2.LINE_AA)
    return img


def create_directory_card(directory):
    path = Path(directory)
    img = np.full((560, 400, 3), (22, 30, 38), np.uint8)
    cv2.rectangle(img, (0, 0), (400, 82), (62, 88, 60), -1)
    cv2.putText(img, "DEVICE FOLDER", (20, 52), FONT, .78, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "[ DIR ]", (24, 200), FONT, 1.5, TECH_ACCENT, 3, cv2.LINE_AA)
    name = path.name or str(path)
    for row, line in enumerate(wrap_label(name, 22)):
        cv2.putText(img, line, (24, 272 + row * 40), FONT, .8, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "HOLD POINT TO ENTER", (24, 512), FONT, .6, TECH_BLUE, 1, cv2.LINE_AA)
    return img


def create_drive_card(drive):
    path = Path(drive)
    img = np.full((560, 400, 3), (18, 28, 40), np.uint8)
    cv2.rectangle(img, (0, 0), (400, 82), (70, 60, 110), -1)
    cv2.putText(img, "STORAGE DRIVE", (20, 52), FONT, .78, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "[ DRIVE ]", (24, 200), FONT, 1.25, TECH_ACCENT, 3, cv2.LINE_AA)
    for row, line in enumerate(wrap_label(str(path), 20)):
        cv2.putText(img, line, (24, 272 + row * 40), FONT, .82, TECH_WHITE, 2, cv2.LINE_AA)
    try:
        usage = os.statvfs(path)
        cv2.putText(img, f"{usage.f_bavail * usage.f_frsize / (1024 ** 3):,.0f} GB FREE",
                    (24, 452), FONT, .62, TECH_BLUE, 1, cv2.LINE_AA)
    except (OSError, AttributeError):
        pass
    cv2.putText(img, "HOLD POINT TO OPEN", (24, 512), FONT, .6, TECH_BLUE, 1, cv2.LINE_AA)
    return img


def list_drives():
    """Enumerate real storage roots: drive letters on Windows, mounts on Unix."""
    system, drives = platform.system(), []
    if system == "Windows":
        import string
        from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                drives.append(Path(f"{letter}:\\"))
    else:
        drives.append(Path("/"))
        for base in ("/mnt", "/media", "/Volumes", "/run/media"):
            root = Path(base)
            if root.is_dir():
                for entry in sorted(root.iterdir()):
                    try:
                        if entry.is_dir():
                            drives.append(entry)
                    except OSError:
                        continue
        home = Path.home()
        if home not in drives:
            drives.append(home)
    return drives


def read_file_text(file_path):
    path, suffix = Path(file_path), Path(file_path).suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".json", ".csv", ".log", ".xml", ".html", ".js", ".css"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".docx":
            with zipfile.ZipFile(path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
            return "\n".join(node.text or "" for node in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages[:3])
            except ImportError:
                return "PDF reading needs the optional pypdf package. Install it with: pip install pypdf"
        return f"No in-app text reader is available for {suffix or 'this file type'}."
    except Exception as error:
        return f"Could not read this file: {error}"


def create_document_texture(file_path):
    """Render supported file content into the workspace at full fidelity."""
    path = Path(file_path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if photo is not None:
            longest = max(photo.shape[:2])
            if longest > VIEW_MAX_DIM:
                factor = VIEW_MAX_DIM / longest
                photo = cv2.resize(photo, (int(photo.shape[1] * factor), int(photo.shape[0] * factor)),
                                   interpolation=cv2.INTER_AREA)
            return photo
        return create_file_preview(path)
    img = np.full((1400, 1000, 3), (238, 240, 235), np.uint8)
    cv2.rectangle(img, (0, 0), (1000, 96), (42, 78, 98), -1)
    cv2.putText(img, path.name[:58], (28, 60), FONT, .85, TECH_WHITE, 2, cv2.LINE_AA)
    lines, y = read_file_text(path).splitlines() or ["(empty file)"], 150
    for line in lines[:40]:
        clean = line.encode("ascii", "replace").decode("ascii")
        cv2.putText(img, clean[:92], (30, y), FONT, .6, (34, 40, 45), 1, cv2.LINE_AA)
        y += 31
    cv2.putText(img, "IN-APP READ MODE", (30, 1360), FONT, .55, TECH_BLUE_DARK, 1, cv2.LINE_AA)
    return img


def load_directory(carousel, directory):
    folder = Path(directory)
    carousel.reset()
    try:
        entries = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError as error:
        print(f"[JARVIS] Cannot browse {folder}: {error}"); return folder
    if folder.parent != folder:
        carousel.add_file(create_directory_card(folder.parent), str(folder.parent))
    for entry in entries[:14]:
        carousel.add_file(create_file_preview(entry), str(entry))
    return folder


def load_drives(carousel):
    carousel.reset()
    for drive in list_drives():
        carousel.add_file(create_drive_card(drive), str(drive))


def draw_browse_computer_action(img, pointer):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 28, int(h * .16), min(w - 28, 245), int(h * .29)
    focused = pointer and x1 <= pointer[0] <= x2 and y1 <= pointer[1] <= y2
    color = TECH_ACCENT if focused else TECH_BLUE
    translucent_rect(img, (x1, y1), (x2, y2), (12, 26, 38), .78)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if focused else 1, cv2.LINE_AA)
    glow_text(img, "EXPLORE DEVICE", (x1 + 14, y1 + 31), .43, color, 1)
    glow_text(img, "POINT + HOLD TO BROWSE", (x1 + 14, y1 + 53), .31, TECH_WHITE, 1)
    return bool(focused)


# ---------------------------------------------------------------------------
# Document scanner ("file upload")
# ---------------------------------------------------------------------------
def order_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def enhance_scan(img):
    """Give a captured page a clean, evenly-lit 'scanned' studio look."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    luma = cv2.createCLAHE(2.2, (8, 8)).apply(luma)
    merged = cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.addWeighted(merged, 1.5, cv2.GaussianBlur(merged, (0, 0), 1.2), -0.5, 0)


def warp_document(frame, quad):
    tl, tr, br, bl = order_points(quad)
    max_w = max(200, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    max_h = max(200, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.float32([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]])
    matrix = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    return enhance_scan(cv2.warpPerspective(frame, matrix, (max_w, max_h)))


class DocumentScanner:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.pages, self.last_quad = [], None
        self.flash_at, self.toast, self.toast_at = 0.0, "", 0.0

    def set_toast(self, message):
        self.toast, self.toast_at = message, time.time()

    def detect(self, frame):
        """Find the largest convex 4-point contour = the page held to camera."""
        h, w = frame.shape[:2]
        scale = w / DETECT_WIDTH if w > DETECT_WIDTH else 1.0
        small = cv2.resize(frame, (int(w / scale), int(h / scale))) if scale != 1.0 else frame
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.dilate(cv2.Canny(gray, 60, 180), np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        area = small.shape[0] * small.shape[1]
        self.last_quad = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4 and cv2.contourArea(approx) > 0.18 * area and cv2.isContourConvex(approx):
                self.last_quad = order_points(approx.reshape(4, 2).astype(np.float32) * scale)
                break
        return self.last_quad

    def capture(self, frame):
        if self.last_quad is None:
            self.set_toast("NO PAGE DETECTED - ALIGN THE DOCUMENT")
            return False
        self.pages.append(warp_document(frame, self.last_quad))
        self.flash_at = time.time()
        self.set_toast(f"CAPTURED PAGE {len(self.pages)}")
        return True

    def save(self):
        if not self.pages:
            self.set_toast("NO PAGES CAPTURED YET")
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"scan_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
        try:
            from PIL import Image
            frames = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in self.pages]
            frames[0].save(str(path), "PDF", save_all=True, append_images=frames[1:], resolution=150.0)
            self.set_toast(f"SAVED {len(self.pages)} PAGE PDF -> {path}")
            self.pages = []
        except ImportError:
            folder = path.with_suffix("")
            folder.mkdir(parents=True, exist_ok=True)
            for i, page in enumerate(self.pages):
                cv2.imwrite(str(folder / f"page_{i + 1:02d}.png"), page)
            self.set_toast(f"Pillow missing - saved PNGs -> {folder} (pip install Pillow for PDF)")
            self.pages = []
        except Exception as error:
            self.set_toast(f"SAVE FAILED: {error}")


def draw_scan(img, scanner, pointer, thumb_progress, t):
    """Live scanner overlay. Returns the button key under the pointer, if any."""
    h, w = img.shape[:2]
    translucent_rect(img, (0, 0), (w, 54), HUD_BG, .55)
    translucent_rect(img, (0, h - 38), (w, h), HUD_BG, .55)
    quad = scanner.last_quad
    if quad is not None:
        pts = np.int32(quad)
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, pts, (40, 90, 40))
        cv2.addWeighted(overlay, .25, img, .75, 0, img)
        cv2.polylines(img, [pts], True, TECH_GREEN, 3, cv2.LINE_AA)
        for point in pts:
            cv2.circle(img, tuple(point), 8, TECH_ACCENT, -1, cv2.LINE_AA)
        glow_text(img, "PAGE LOCKED - THUMBS-UP TO CAPTURE", (w // 2, int(h * .12)), .6, TECH_GREEN, 2, True)
    else:
        glow_text(img, "SHOW A PAGE / DOCUMENT TO THE CAMERA", (w // 2, int(h * .12)), .6, TECH_ACCENT, 2, True)
    if thumb_progress > 0:
        bar_w = min(360, int(w * .4)); x1 = w // 2 - bar_w // 2; y = int(h * .17)
        cv2.rectangle(img, (x1, y), (x1 + bar_w, y + 10), TECH_BLUE_DARK, 1)
        cv2.rectangle(img, (x1, y), (x1 + int(bar_w * thumb_progress), y + 10), TECH_ACCENT, -1)
    glow_text(img, f"PAGES: {len(scanner.pages)}", (w - 168, 82), .55, TECH_WHITE, 1)
    for i, page in enumerate(scanner.pages[-6:]):
        y0 = 100 + i * 132
        img[y0:y0 + 120, w - 112:w - 22] = cv2.resize(page, (90, 120))
        cv2.rectangle(img, (w - 112, y0), (w - 22, y0 + 120), TECH_BLUE, 1, cv2.LINE_AA)
    hovered = None
    bw, bh, gap = 220, 62, 44
    bx = w // 2 - (bw * 2 + gap) // 2; by = h - 116
    for idx, (label, key, enabled) in enumerate([("SAVE PDF", "scan:save", bool(scanner.pages)),
                                                 ("DISCARD", "scan:discard", bool(scanner.pages))]):
        x1 = bx + idx * (bw + gap); x2 = x1 + bw; y1, y2 = by, by + bh
        inside = pointer and x1 <= pointer[0] <= x2 and y1 <= pointer[1] <= y2 and enabled
        color = TECH_ACCENT if inside else (TECH_BLUE if enabled else TECH_BLUE_DARK)
        translucent_rect(img, (x1, y1), (x2, y2), (10, 22, 34), .75)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if inside else 1, cv2.LINE_AA)
        glow_text(img, label, (x1 + bw // 2, y1 + 40), .62, color, 2, True)
        if inside:
            hovered = key
    if scanner.toast and t - scanner.toast_at < 5:
        glow_text(img, scanner.toast[:74], (w // 2, h - 54), .48, TECH_WHITE, 1, True)
    if t - scanner.flash_at < 0.25:
        translucent_rect(img, (0, 0), (w, h), (255, 255, 255), .6 * (1 - (t - scanner.flash_at) / 0.25))
    return hovered


# ---------------------------------------------------------------------------
# Carousel + workspace + selection
# ---------------------------------------------------------------------------
class DwellSelector:
    """Point-and-hold selection: unambiguous, no finger-distance thresholds.

    Fires once when the pointer stays near a stable target for ``hold`` seconds.
    Moving away, or the target changing, resets the timer so it can fire again.
    """
    def __init__(self, hold=1.05, radius=52, cooldown=0.7):
        self.hold, self.radius, self.cooldown = hold, radius, cooldown
        self.anchor, self.start, self.key, self.done = None, None, None, False
        self.blocked_until = 0.0

    def reset(self):
        self.anchor, self.start, self.key, self.done = None, None, None, False

    def block(self, now, seconds=None):
        """Freeze selection for a moment - e.g. right after entering a folder."""
        self.blocked_until = now + (self.cooldown if seconds is None else seconds)
        self.reset()

    def update(self, point, key, now):
        if now < self.blocked_until or point is None or key is None:
            self.reset()
            return 0.0, False
        moved = self.anchor is not None and math.dist(point, self.anchor) > self.radius
        if self.key != key or self.anchor is None or moved:
            self.anchor, self.start, self.key, self.done = list(point), now, key, False
        progress = min(1.0, (now - self.start) / self.hold)
        if progress >= 1.0 and not self.done:
            self.done = True
            self.blocked_until = now + self.cooldown        # cooldown before next select
            return 1.0, True
        return progress, False


class CarouselEngine:
    def __init__(self):
        self.files, self.file_paths, self.scroll_float = [], [], 0.0
        self.offset, self.scale = [0.0, 0.0], 1.0
    def reset(self):
        self.files, self.file_paths, self.scroll_float = [], [], 0.0
        self.offset, self.scale = [0.0, 0.0], 1.0
    def add_file(self, texture, file_path=None):
        self.files.append(texture); self.file_paths.append(file_path)
    def focused_index(self): return int(round(self.scroll_float)) if self.files else -1
    def settled(self):
        index = self.focused_index()
        return index >= 0 and abs(self.scroll_float - index) < 0.28
    def focused_path(self):
        index = self.focused_index()
        return self.file_paths[index] if 0 <= index < len(self.file_paths) else None
    def label_for(self, index):
        if 0 <= index < len(self.file_paths) and self.file_paths[index]:
            path = Path(self.file_paths[index])
            return path.name or str(path)
        return None
    def update(self, x, width):
        if x is None or not self.files: return
        target = np.clip((x - .20*width) / max(1, .60*width), 0, 1) * (len(self.files)-1)
        self.scroll_float += (target - self.scroll_float) * .2
    def manipulate(self, midpoint, span, anchor):
        if midpoint is None or span is None:
            return None
        if anchor is None:
            anchor = (midpoint, span, list(self.offset), self.scale)
        a_mid, a_span, a_off, a_scale = anchor
        self.offset = [a_off[0] + (midpoint[0]-a_mid[0]), a_off[1] + (midpoint[1]-a_mid[1])]
        self.scale = float(np.clip(a_scale * span/max(1.0, a_span), 0.55, 2.4))
        return anchor
    def draw(self, img):
        h, w = img.shape[:2]
        if not self.files:
            glow_text(img, "// ARCHIVE EMPTY", (w//2, h//2), .7, centered=True); return img
        base_h = int(h*.40*self.scale); base_w = int(base_h*.72); spacing = int(base_h*1.02)
        cx0 = w//2 + int(self.offset[0]); cy0 = int(h*.50) + int(self.offset[1])
        focus = self.focused_index()
        for i in sorted(range(len(self.files)), key=lambda n: abs(n-self.scroll_float), reverse=True):
            diff, distance = i-self.scroll_float, abs(i-self.scroll_float)
            scale = max(.45, 1-distance*.22); pw, ph = int(base_w*scale), int(base_h*scale)
            cx, cy = cx0 + int(diff*spacing), cy0 + int(distance*18)
            pts = [[cx-pw//2, cy-ph//2], [cx+pw//2, cy-ph//2], [cx+pw//2, cy+ph//2], [cx-pw//2, cy+ph//2]]
            img, poly = paste_texture(img, self.files[i], pts, max(.35, 1-distance*.38))
            cv2.polylines(img, [poly], True, TECH_ACCENT if i == focus else TECH_BLUE_DARK, 3 if i == focus else 1, cv2.LINE_AA)
            if i == focus: glow_text(img, f"[ {i+1:02d} / {len(self.files):02d} ]", (cx-pw//2, cy-ph//2-14), .6, TECH_WHITE)
        name = self.label_for(focus)
        if name:
            banner_y = min(h-24, cy0 + int(base_h*.5) + 46)
            translucent_rect(img, (0, banner_y-40), (w, banner_y+16), (10, 22, 34), .72)
            glow_text(img, name[:52], (w//2, banner_y), .82, TECH_WHITE, 2, True)
        return img


class HoloPanelEngine:
    def __init__(self): self.center, self.size, self.texture, self.is_active = None, 280, None, False
    def draw(self, img, t):
        if not self.is_active or self.texture is None: return img
        h, w = img.shape[:2]; self.size = int(np.clip(self.size, 120, min(w, h)-40))
        if self.center is None: self.center = [w//2, h//2]
        half = self.size//2; self.center = [int(np.clip(self.center[0], half, w-half)), int(np.clip(self.center[1], half, h-half))]
        x, y = self.center; aspect = self.texture.shape[0]/self.texture.shape[1]; hh = int(half*aspect)
        img, poly = paste_texture(img, self.texture, [[x-half,y-hh],[x+half,y-hh],[x+half,y+hh],[x-half,y+hh]])
        cv2.polylines(img, [poly], True, TECH_BLUE_DARK, 6, cv2.LINE_AA); cv2.polylines(img, [poly], True, TECH_BLUE, 2, cv2.LINE_AA)
        glow_text(img, "// ACTIVE DOCUMENT", (x-half, y-hh-12), .5)
        return img


def hand_scale(lm): return math.hypot(lm[0][1]-lm[9][1], lm[0][2]-lm[9][2]) + 1e-6
def is_pinching(lm):
    if not lm: return False, None
    dist = math.hypot(lm[4][1]-lm[8][1], lm[4][2]-lm[8][2])
    return dist < .55*hand_scale(lm), ((lm[4][1]+lm[8][1])//2, (lm[4][2]+lm[8][2])//2)
def is_palm_open(lm):
    if not lm or is_pinching(lm)[0]: return False
    wx, wy = lm[0][1], lm[0][2]
    return sum(math.hypot(lm[t][1]-wx,lm[t][2]-wy) > math.hypot(lm[p][1]-wx,lm[p][2]-wy)*1.15 for t,p in [(8,6),(12,10),(16,14),(20,18)]) == 4
def is_closed_fist(lm):
    if not lm: return False
    wx, wy, scale = lm[0][1], lm[0][2], hand_scale(lm)
    return sum(math.hypot(lm[tip][1]-wx, lm[tip][2]-wy) < 1.55 * scale for tip in (8, 12, 16, 20)) >= 4
def is_thumbs_up(lm):
    if not lm: return False
    scale = hand_scale(lm)
    thumb_is_up = lm[4][2] < lm[2][2] - .65 * scale
    fingers_folded = sum(lm[tip][2] > lm[pip][2] - .15 * scale for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)])
    return thumb_is_up and fingers_folded >= 3
def is_scissor(lm):
    """Thumb + index snipped together while the other fingers stay extended -
    a deliberate 'commit / lock' snip, distinct from a full pinch-grab or fist."""
    if not lm: return False
    scale = hand_scale(lm)
    thumb_index = math.hypot(lm[4][1]-lm[8][1], lm[4][2]-lm[8][2])
    wx, wy = lm[0][1], lm[0][2]
    extended = sum(math.hypot(lm[t][1]-wx, lm[t][2]-wy) > math.hypot(lm[p][1]-wx, lm[p][2]-wy)*1.15
                   for t, p in [(12, 10), (16, 14), (20, 18)])
    return thumb_index < 0.5 * scale and extended >= 2
def get_hands(results, w, h):
    hands = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hands[handedness.classification[0].label] = [[i, int(p.x*w), int(p.y*h)] for i,p in enumerate(landmarks.landmark)]
    return hands
def smooth_point(old, new, alpha=.5):
    return list(new) if old is None else [old[0]+(new[0]-old[0])*alpha, old[1]+(new[1]-old[1])*alpha]


class VisualClapDetector:
    def __init__(self): self.closed_at = None
    def update(self, left, right, now):
        if not left or not right:
            if self.closed_at and now-self.closed_at > 1.2: self.closed_at = None
            return False
        lc, rc = np.array(left[9][1:]), np.array(right[9][1:])
        spacing = np.linalg.norm(lc-rc) / ((hand_scale(left)+hand_scale(right))/2)
        if spacing < 1.25: self.closed_at = now
        elif self.closed_at and now-self.closed_at < .8 and spacing > 2.0:
            self.closed_at = None; return True
        return False
    def prompt(self, left, right):
        if not left or not right: return "SHOW BOTH HANDS TO CONTINUE"
        return "CLAP DETECTED - RELEASE HANDS" if self.closed_at else "CLAP YOUR HANDS TO INITIALIZE"


def draw_welcome(img, t, prompt):
    h, w = img.shape[:2]
    translucent_rect(img, (0,0), (w,h), (4,8,12), .68)
    radius = int(min(w,h)*.115 + 6*math.sin(t*2))
    cv2.circle(img, (w//2, int(h*.38)), radius, TECH_BLUE_DARK, 2, cv2.LINE_AA)
    cv2.circle(img, (w//2, int(h*.38)), radius-22, TECH_BLUE, 1, cv2.LINE_AA)
    for a in range(0, 360, 45):
        rad = math.radians(a+t*55); r = radius+16
        pt = (int(w/2+math.cos(rad)*r), int(h*.38+math.sin(rad)*r))
        cv2.circle(img, pt, 3, TECH_ACCENT, -1)
    glow_text(img, "J.A.R.V.I.S.", (w//2, int(h*.60)), .95, TECH_BLUE, 2, True)
    glow_text(img, WELCOME_TEXT, (w//2, int(h*.68)), .62, TECH_WHITE, 2, True)
    glow_text(img, "WHAT DO WE DO TODAY?", (w//2, int(h*.73)), .46, TECH_BLUE, 1, True)
    glow_text(img, prompt, (w//2, int(h*.86)), .5, TECH_ACCENT, 1, True)


def draw_main_menu(img, pointer, t):
    h, w = img.shape[:2]; cx, cy = w//2, h//2
    translucent_rect(img, (0,0), (w,h), (4,8,12), .42)
    glow_text(img, "MAIN MENU", (cx, int(h*.15)), .68, TECH_BLUE, 2, True)
    glow_text(img, "POINT TO FOCUS  |  HOLD TO SELECT", (cx, int(h*.20)), .40, TECH_WHITE, 1, True)
    cards = [("01", "WORKSPACE", "Browse and manipulate active documents", True),
             ("02", "SCANNER", "Scan pages into a saved PDF", True),
             ("03", "COMMS", "Reserved for future modules", False)]
    card_w, card_h, gap = min(360, int(w*.55)), 64, 28
    hovered = None
    for i, (num, title, subtitle, enabled) in enumerate(cards):
        y1 = cy - card_h - gap + i*(card_h+gap); y2 = y1+card_h; x1, x2 = cx-card_w//2, cx+card_w//2
        inside = pointer and x1 <= pointer[0] <= x2 and y1 <= pointer[1] <= y2
        color = TECH_ACCENT if inside and enabled else (TECH_BLUE if enabled else TECH_BLUE_DARK)
        translucent_rect(img, (x1,y1), (x2,y2), (9,20,30), .72)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2 if inside else 1, cv2.LINE_AA)
        glow_text(img, num, (x1+16, y1+29), .55, color, 2)
        glow_text(img, title, (x1+66, y1+27), .50, TECH_WHITE if enabled else TECH_BLUE_DARK, 2)
        glow_text(img, subtitle, (x1+66, y1+49), .34, color)
        if inside and enabled: hovered = title
    return hovered


def draw_navigation_hint(img, mode):
    h, w = img.shape[:2]
    trail = {"MENU": "MENU",
             "SCANNER": "MENU / SCANNER",
             "CAROUSEL": "MENU / WORKSPACE / ARCHIVE",
             "DRIVE_SELECT": "MENU / WORKSPACE / DEVICE / DRIVE",
             "DEVICE_BROWSER": "MENU / WORKSPACE / DEVICE / DRIVE / FOLDER",
             "WORKSPACE": "MENU / WORKSPACE / ARCHIVE / DOCUMENT"}[mode]
    glow_text(img, trail, (24, 24), .42, TECH_BLUE, 1)
    glow_text(img, "HOLD FIST: BACK   |   HOLD POINT: SELECT   |   m: MENU", (w-430, 24), .38, TECH_WHITE, 1)


def draw_open_confirmation(img, file_path, thumb_progress):
    h, w = img.shape[:2]
    translucent_rect(img, (0, 0), (w, h), (4, 8, 12), .62)
    glow_text(img, "OPEN FILE?", (w//2, int(h*.37)), .9, TECH_BLUE, 2, True)
    glow_text(img, Path(file_path).name[:48], (w//2, int(h*.45)), .52, TECH_WHITE, 1, True)
    glow_text(img, "GIVE A THUMBS-UP AND HOLD TO CONFIRM", (w//2, int(h*.57)), .46, TECH_ACCENT, 1, True)
    glow_text(img, "HOLD A CLOSED FIST TO CANCEL", (w//2, int(h*.63)), .36, TECH_WHITE, 1, True)
    bar_w, x1, y = min(340, int(w*.45)), w//2-min(340, int(w*.45))//2, int(h*.69)
    cv2.rectangle(img, (x1, y), (x1+bar_w, y+10), TECH_BLUE_DARK, 1)
    cv2.rectangle(img, (x1, y), (x1+int(bar_w*thumb_progress), y+10), TECH_ACCENT, -1)


def open_camera():
    api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    working = []
    print("[JARVIS] Scanning camera interfaces...")
    for index in range(10):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            time.sleep(.1); ok, frame = cap.read()
            if ok and frame is not None: working.append(index); print(f"[JARVIS] Camera {index}: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    if not working: return cv2.VideoCapture()
    selected = next((i for i in working if i != 0), working[0])
    cap = cv2.VideoCapture(selected, api)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[JARVIS] Camera {selected} selected for high-resolution viewer.")
    return cap


def main():
    cap = open_camera()
    if not cap.isOpened(): print("[JARVIS] ERROR: could not open a camera."); return
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=.7, min_tracking_confidence=.6)
    workspace, carousel, clap = HoloPanelEngine(), CarouselEngine(), VisualClapDetector()
    display_filter, device_directory = StudioCameraFilter(), Path.home()
    scanner, dwell = DocumentScanner(SCAN_DIR), DwellSelector()
    mode, previous_pinch, grabbed, reticle_pt, fullscreen = "WELCOME", False, False, None, False
    resize_anchor, fist_started, fist_latched = None, None, False
    pending_file, thumbs_started = None, None
    carousel_anchor, hud_zoom, hud_zoom_anchor = None, 1.0, None
    manip_locked, lock_toast_at = False, 0.0
    scan_thumb_started, scan_thumb_latched = None, False
    zoom_modes = {"MENU", "SCANNER", "CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER"}
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO); cv2.resizeWindow(WIN, 1280, 720)
    fps, last, frames = 0., time.time(), 0
    while True:
        ok, raw = cap.read()
        if not ok: break
        t = time.time(); raw = cv2.flip(raw, 1)
        # Raw feeds MediaPipe / scanning; only the viewer gets the studio grade.
        img = display_filter.apply(raw); h, w = img.shape[:2]
        tracking = low_quality_tracking_frame(raw)
        results = detector.process(cv2.cvtColor(tracking, cv2.COLOR_BGR2RGB))
        hands = get_hands(results, w, h); left, right = hands.get("Left"), hands.get("Right"); pointer = right or left
        p_pinch, p_pt = is_pinching(pointer) if pointer else (False, None); l_pinch, l_pt = is_pinching(left); r_pinch, r_pt = is_pinching(right)
        pointer_pt = (pointer[8][1], pointer[8][2]) if pointer else None
        reticle_pt = smooth_point(reticle_pt, pointer_pt) if pointer_pt else None
        pinch_down = p_pinch and not previous_pinch
        two_hand_pinch = l_pinch and r_pinch
        both_palms_open = is_palm_open(left) and is_palm_open(right)
        interacting = two_hand_pinch or both_palms_open        # suppress selection/scroll
        closed_fist = any(is_closed_fist(hand) for hand in hands.values())
        if closed_fist:
            fist_started = t if fist_started is None else fist_started
        else:
            fist_started, fist_latched = None, False
        back_requested = (mode in {"CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER", "WORKSPACE", "CONFIRM_OPEN", "SCANNER"}
                          and not fist_latched and fist_started is not None and t - fist_started >= .60)
        if back_requested:
            fist_latched = True; dwell.reset()
            if mode == "WORKSPACE": mode, workspace.is_active, grabbed = "CAROUSEL", False, False
            elif mode == "CAROUSEL": mode = "MENU"
            elif mode == "SCANNER": mode = "MENU"
            elif mode == "DRIVE_SELECT": mode = "CAROUSEL"
            elif mode == "DEVICE_BROWSER": mode = "DRIVE_SELECT"; load_drives(carousel)
            elif mode == "CONFIRM_OPEN": mode, pending_file, thumbs_started = "DEVICE_BROWSER", None, None

        # Global HUD zoom: two-hand pinch spread / squeeze (freed up by dwell).
        if two_hand_pinch and mode in zoom_modes and l_pt and r_pt:
            span = math.dist(l_pt, r_pt)
            if hud_zoom_anchor is None: hud_zoom_anchor = (span, hud_zoom)
            a_span, a_zoom = hud_zoom_anchor
            hud_zoom = float(np.clip(a_zoom * span / max(1.0, a_span), 0.6, 2.2))
        else:
            hud_zoom_anchor = None

        # Two open palms drag + scale the whole archive; a thumb+index SNIP
        # (scissor) commits the placement and locks it so lowering your hands
        # can't disturb it. The lock clears once both hands are fully released.
        scissor = any(is_scissor(hand) for hand in hands.values())
        if scissor and mode in {"CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER"}:
            if not manip_locked: lock_toast_at = t
            manip_locked, carousel_anchor = True, None
        if not both_palms_open and not scissor:
            manip_locked = False
        carousel_manip = both_palms_open and not manip_locked and mode in {"CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER"}
        if carousel_manip:
            midpoint = ((left[9][1]+right[9][1])//2, (left[9][2]+right[9][2])//2)
            carousel_anchor = carousel.manipulate(midpoint, math.dist(left[9][1:], right[9][1:]), carousel_anchor)
        else:
            carousel_anchor = None

        def browse_scroll():
            if not interacting:
                carousel.update(pointer[8][1] if pointer else None, w)

        if mode == "WELCOME":
            if clap.update(left, right, t): mode = "MENU"
            draw_welcome(img, t, clap.prompt(left, right))
        elif mode == "MENU":
            hover = draw_main_menu(img, reticle_pt, t)
            draw_navigation_hint(img, "MENU")
            key = f"menu:{hover}" if hover and not interacting else None
            progress, fired = dwell.update(reticle_pt, key, t)
            draw_dwell_ring(img, reticle_pt, progress)
            if fired:
                dwell.reset()
                if hover == "WORKSPACE": mode = "CAROUSEL"
                elif hover == "SCANNER": mode = "SCANNER"
        elif mode == "SCANNER":
            scanner.detect(raw)
            thumbs = any(is_thumbs_up(hand) for hand in hands.values())
            if thumbs and scan_thumb_started is None: scan_thumb_started = t
            if not thumbs: scan_thumb_started, scan_thumb_latched = None, False
            thumb_progress = min(1.0, (t - scan_thumb_started) / .4) if scan_thumb_started else 0.0
            if thumb_progress >= 1.0 and not scan_thumb_latched:
                scan_thumb_latched = True; scanner.capture(raw)
            hover_key = draw_scan(img, scanner, reticle_pt, thumb_progress, t)
            draw_navigation_hint(img, "SCANNER")
            key = hover_key if hover_key and not interacting else None
            progress, fired = dwell.update(reticle_pt, key, t)
            draw_dwell_ring(img, reticle_pt, progress)
            if fired:
                dwell.reset()
                if hover_key == "scan:save": scanner.save()
                elif hover_key == "scan:discard": scanner.pages = []; scanner.set_toast("PAGES DISCARDED")
            reticle(img, reticle_pt, t)
        elif mode == "CONFIRM_OPEN":
            thumbs_up = any(is_thumbs_up(hand) for hand in hands.values())
            thumbs_started = t if thumbs_up and thumbs_started is None else thumbs_started
            if not thumbs_up: thumbs_started = None
            progress = min(1.0, (t - thumbs_started) / .45) if thumbs_started else 0.0
            draw_open_confirmation(img, pending_file, progress)
            if progress >= 1.0:
                workspace.texture, workspace.is_active = create_document_texture(pending_file), True
                workspace.center, workspace.size = [w//2, h//2], int(min(w, h)*.7)
                mode, pending_file, thumbs_started = "WORKSPACE", None, None
        else:
            translucent_rect(img, (0,0), (w,54), HUD_BG, .55); translucent_rect(img, (0,h-38), (w,h), HUD_BG, .55); corner_brackets(img)
            if carousel_manip:
                glow_text(img, f"MOVE + SCALE ARCHIVE  x{carousel.scale:0.2f}  (SNIP TO LOCK)", (w//2, 48), .5, TECH_ACCENT, 1, True)
            elif t - lock_toast_at < 1.4:
                glow_text(img, "ARCHIVE LOCKED - SNIP RECOGNIZED", (w//2, 48), .5, TECH_GREEN, 2, True)
            if mode == "CAROUSEL":
                draw_navigation_hint(img, "CAROUSEL")
                glow_text(img, "HOLD POINT: EXPLORE DEVICE   |   BOTH PALMS: MOVE/RESIZE   |   TWO-HAND PINCH: HUD ZOOM", (24,48), .32, TECH_WHITE)
                browse_scroll(); img = carousel.draw(img)
                browse_focused = draw_browse_computer_action(img, reticle_pt)
                key = "browse" if browse_focused and not interacting else None
                progress, fired = dwell.update(reticle_pt, key, t); draw_dwell_ring(img, reticle_pt, progress)
                if fired: dwell.reset(); load_drives(carousel); mode = "DRIVE_SELECT"
            elif mode == "DRIVE_SELECT":
                draw_navigation_hint(img, "DRIVE_SELECT")
                glow_text(img, "STEP 1/3: PICK A DRIVE   |   HOLD POINT: OPEN   |   BOTH PALMS: MOVE/RESIZE", (24,48), .34, TECH_WHITE)
                browse_scroll(); img = carousel.draw(img)
                focused_path = carousel.focused_path()
                key = ("card", carousel.focused_index()) if focused_path and carousel.settled() and not interacting else None
                progress, fired = dwell.update(reticle_pt, key, t); draw_dwell_ring(img, reticle_pt, progress)
                if fired and focused_path:
                    dwell.reset(); device_directory = load_directory(carousel, focused_path); mode = "DEVICE_BROWSER"
            elif mode == "DEVICE_BROWSER":
                draw_navigation_hint(img, "DEVICE_BROWSER")
                glow_text(img, f"STEP 2-3: {str(device_directory)[:52]}   |   HOLD POINT: OPEN FOLDER / FILE", (24,48), .34, TECH_WHITE)
                browse_scroll(); img = carousel.draw(img)
                focused_path = carousel.focused_path()
                key = ("card", carousel.focused_index()) if focused_path and carousel.settled() and not interacting else None
                progress, fired = dwell.update(reticle_pt, key, t); draw_dwell_ring(img, reticle_pt, progress)
                if fired and focused_path:
                    dwell.reset()
                    if Path(focused_path).is_dir(): device_directory = load_directory(carousel, focused_path)
                    elif Path(focused_path).is_file(): pending_file, thumbs_started, mode = focused_path, None, "CONFIRM_OPEN"
            elif mode == "WORKSPACE":
                draw_navigation_hint(img, "WORKSPACE")
                glow_text(img, "PINCH+DRAG: MOVE   |   PINCH BOTH HANDS, THEN SPREAD / SQUEEZE: RESIZE", (24,48), .34, TECH_WHITE)
                if l_pinch and r_pinch and l_pt and r_pt:
                    span = max(1.0, math.dist(l_pt, r_pt))
                    if resize_anchor is None: resize_anchor = (span, workspace.size)
                    start_span, start_size = resize_anchor
                    workspace.size = int(np.clip(start_size * span / start_span, 100, min(w,h)-40)); grabbed = False
                    cv2.line(img, l_pt, r_pt, TECH_ACCENT, 1, cv2.LINE_AA)
                    glow_text(img, f"RESIZE {workspace.size}px", ((l_pt[0]+r_pt[0])//2, (l_pt[1]+r_pt[1])//2-12), .38, TECH_ACCENT, 1, True)
                elif p_pinch and p_pt:
                    resize_anchor = None
                    cx, cy, half = workspace.center[0], workspace.center[1], workspace.size//2
                    if pinch_down and cx-half < p_pt[0] < cx+half and cy-half < p_pt[1] < cy+half: grabbed = True
                    if grabbed: workspace.center = list(p_pt)
                else: grabbed, resize_anchor = False, None
                img = workspace.draw(img, t)
            reticle(img, reticle_pt, t)
        frames += 1
        if t-last >= .5: fps, frames, last = frames/(t-last), 0, t
        glow_text(img, f"VISION: {len(hands)} HANDS | {fps:4.1f} FPS | FILTER: {display_filter.name()} | HUD x{hud_zoom:0.2f}", (20,h-14), .36, TECH_BLUE)
        glow_text(img, "m MENU  c FILTER  +/- ZOOM  f FULL  q QUIT", (w-320,h-14), .36, TECH_BLUE_DARK)
        previous_pinch = p_pinch
        cv2.imshow(WIN, apply_hud_zoom(img, hud_zoom))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key == ord('m') and mode != "WELCOME":
            mode, workspace.is_active, grabbed = "MENU", False, False; dwell.reset()
        if key == ord('c'): display_filter.cycle()
        if key in (ord('+'), ord('=')): hud_zoom = min(2.2, hud_zoom + 0.1)
        if key == ord('-'): hud_zoom = max(0.6, hud_zoom - 0.1)
        if key == ord('0'): hud_zoom = 1.0
        if key == ord('f'):
            fullscreen = not fullscreen; cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break
    cap.release(); detector.close(); cv2.destroyAllWindows()


if __name__ == "__main__": main()
