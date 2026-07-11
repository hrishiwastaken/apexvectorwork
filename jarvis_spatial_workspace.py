"""J.A.R.V.I.S. Spatial Workspace - camera-first, gesture-controlled HUD.

Controls: hold a peace sign to dismiss the welcome screen; point at a card
and pinch-hold to select/focus/open (works for both folders and files).
Open BOTH palms over the archive to drag it around, and spread / squeeze both
palms to resize it. Hold a closed fist to go back up one layer.

Workspace toolbar (left rail): a FOLDER button to browse the device, a SCAN
button to capture a physical notebook as a PDF (just hold a notebook up to the
camera), a DRAW button to sketch on screen with a pinch, and a CURSOR button
that hides the window so you can drive the OS mouse pointer with your hand.

In Workspace (Document View): global gestures are disabled. Use the holographic
[X] button to close the document. Pinch the document to move it, two-hand pinch
to resize it.
"""
import cv2
import math
import platform
import time
from datetime import datetime
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
TECH_RED = (70, 70, 235)
HUD_BG = (35, 20, 5)
DETECT_WIDTH = 640
WELCOME_TEXT = "WELCOME BACK, HRISHI"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_MAX_DIM = 2600
DRAW_COLOR = (255, 210, 40)
SCAN_DIR = Path.home() / "JARVIS_Scans"


class AppConfig:
    def __init__(self):
        self.ui_scale = 1.0
        self.pinch_duration = 0.65


def glow_text(img, text, pos, scale=0.6, color=TECH_BLUE, thick=1, centered=False):
    if centered:
        width = cv2.getTextSize(text, FONT, scale, thick)[0][0]
        pos = (int(pos[0] - width / 2), pos[1])
    x, y = int(pos[0]), int(pos[1])
    cv2.putText(img, text, (x + 2, y + 2), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def wrap_label(text, max_chars):
    words, lines, current = text.replace("_", " _").split(" "), [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current: lines.append(current)
            current = word[:max_chars]
    if current: lines.append(current)
    return lines[:3] or [text[:max_chars]]


def translucent_rect(img, p1, p2, color, alpha):
    x1, y1 = max(0, int(p1[0])), max(0, int(p1[1]))
    x2, y2 = min(img.shape[1], int(p2[0])), min(img.shape[0], int(p2[1]))
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0, roi)


def rounded_panel(img, p1, p2, fill, alpha, border=None, border_thick=1):
    """Cleaner panel primitive with an optional subtle border used across the HUD."""
    translucent_rect(img, p1, p2, fill, alpha)
    if border is not None:
        cv2.rectangle(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), border, border_thick, cv2.LINE_AA)


def corner_brackets(img, margin=20, length=32, color=TECH_BLUE, thick=2):
    h, w = img.shape[:2]
    for p, a, b in [((margin, margin), (margin + length, margin), (margin, margin + length)),
                    ((w-margin, margin), (w-margin-length, margin), (w-margin, margin + length)),
                    ((margin, h-margin), (margin+length, h-margin), (margin, h-margin-length)),
                    ((w-margin, h-margin), (w-margin-length, h-margin), (w-margin, h-margin-length))]:
        cv2.line(img, p, a, color, thick, cv2.LINE_AA)
        cv2.line(img, p, b, color, thick, cv2.LINE_AA)


def reticle(img, point, t, progress_arc=0.0):
    if point is None: return
    x, y = map(int, point)
    r = int(18 + 3 * math.sin(t * 4))
    for k in range(4):
        angle = math.radians((t * 90) + k * 90)
        p1 = (int(x + math.cos(angle)*(r+4)), int(y + math.sin(angle)*(r+4)))
        p2 = (int(x + math.cos(angle)*(r+12)), int(y + math.sin(angle)*(r+12)))
        cv2.line(img, p1, p2, TECH_BLUE, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, TECH_BLUE, 1, cv2.LINE_AA)
    cv2.circle(img, (x, y), 3, TECH_WHITE, -1, cv2.LINE_AA)

    if progress_arc > 0:
        cv2.ellipse(img, (x, y), (r+20, r+20), -90, 0, int(360 * progress_arc), TECH_ACCENT, 3, cv2.LINE_AA)


class StudioCameraFilter:
    def __init__(self): self.previous_clean, self.previous_raw = None, None
    def apply(self, frame):
        spatial = cv2.bilateralFilter(frame, 7, 24, 12)
        if self.previous_clean is not None and self.previous_clean.shape == spatial.shape:
            difference = cv2.cvtColor(cv2.absdiff(frame, self.previous_raw), cv2.COLOR_BGR2GRAY)
            moving = cv2.GaussianBlur((difference > 16).astype(np.uint8) * 255, (0, 0), 2.2)
            stable = cv2.addWeighted(spatial, .62, self.previous_clean, .38, 0)
            mask = cv2.cvtColor(moving, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255
            clean = (spatial * mask + stable * (1 - mask)).astype(np.uint8)
        else:
            clean = spatial
        self.previous_clean, self.previous_raw = clean, frame.copy()
        graded = cv2.convertScaleAbs(clean, alpha=1.07, beta=-3)
        blue, green, red = cv2.split(graded)
        graded = cv2.merge((cv2.add(blue, 5), green, cv2.subtract(red, 2)))
        soft = cv2.GaussianBlur(graded, (0, 0), .65)
        return cv2.addWeighted(graded, 1.08, soft, -.08, 0)


def low_quality_tracking_frame(frame):
    h, w = frame.shape[:2]
    if w <= DETECT_WIDTH: return frame
    # Prevent division by zero if width collapses
    w_safe = max(1, w)
    return cv2.resize(frame, (DETECT_WIDTH, max(1, int(h * DETECT_WIDTH / w_safe))), interpolation=cv2.INTER_AREA)


def paste_texture(img, texture, points, brightness=1.0):
    dst = np.float32(points)
    # Prevent OpenCV Division by Zero / Singular Matrix error if points collapse
    if cv2.contourArea(dst) < 4.0:
        return img, np.int32(dst)

    th, tw = texture.shape[:2]
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]), dst)
    source = cv2.convertScaleAbs(texture, alpha=brightness) if brightness != 1 else texture
    warped = cv2.warpPerspective(source, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    mask = np.zeros(img.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst), (255, 255, 255))
    return cv2.add(cv2.bitwise_and(img, cv2.bitwise_not(mask)), cv2.bitwise_and(warped, mask)), np.int32(dst)


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
    cv2.putText(img, "PINCH & HOLD", (24, 500), FONT, .58, TECH_BLUE, 1, cv2.LINE_AA)
    cv2.putText(img, "TO OPEN", (24, 532), FONT, .5, TECH_BLUE_DARK, 1, cv2.LINE_AA)
    return img


def create_directory_card(directory):
    path = Path(directory)
    img = np.full((560, 400, 3), (22, 30, 38), np.uint8)
    cv2.rectangle(img, (0, 0), (400, 82), (62, 88, 60), -1)
    cv2.putText(img, "DEVICE FOLDER", (20, 52), FONT, .78, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "[ DIR ]", (24, 200), FONT, 1.5, TECH_ACCENT, 3, cv2.LINE_AA)
    for row, line in enumerate(wrap_label(path.name or str(path), 22)):
        cv2.putText(img, line, (24, 272 + row * 40), FONT, .8, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "PINCH & HOLD: ENTER", (24, 512), FONT, .6, TECH_BLUE, 1, cv2.LINE_AA)
    return img


def create_drive_card(drive):
    path = Path(drive)
    img = np.full((560, 400, 3), (18, 28, 40), np.uint8)
    cv2.rectangle(img, (0, 0), (400, 82), (70, 60, 110), -1)
    cv2.putText(img, "STORAGE DRIVE", (20, 52), FONT, .78, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "[ DRIVE ]", (24, 200), FONT, 1.25, TECH_ACCENT, 3, cv2.LINE_AA)
    for row, line in enumerate(wrap_label(str(path), 20)):
        cv2.putText(img, line, (24, 272 + row * 40), FONT, .82, TECH_WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "PINCH & HOLD: OPEN", (24, 512), FONT, .6, TECH_BLUE, 1, cv2.LINE_AA)
    return img


def list_drives():
    system, drives = platform.system(), []
    if system == "Windows":
        import string; from ctypes import windll
        bitmask = windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i): drives.append(Path(f"{letter}:\\"))
    else:
        drives.append(Path("/"))
        for base in ("/mnt", "/media", "/Volumes", "/run/media"):
            root = Path(base)
            if root.is_dir():
                for entry in sorted(root.iterdir()):
                    try:
                        if entry.is_dir(): drives.append(entry)
                    except OSError: continue
        home = Path.home()
        if home not in drives: drives.append(home)
    return drives


def read_file_text(file_path):
    path, suffix = Path(file_path), Path(file_path).suffix.lower()
    try:
        if suffix in {".txt", ".md", ".py", ".json", ".csv", ".log", ".xml", ".html", ".js", ".css"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".docx":
            with zipfile.ZipFile(path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
            return "\n".join(n.text or "" for n in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"))
        if suffix == ".pdf":
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages[:3])
        return f"No in-app text reader is available for {suffix or 'this file type'}."
    except Exception as error: return f"Could not read this file: {error}"


def create_document_texture(file_path):
    path = Path(file_path)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        photo = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if photo is not None:
            longest = max(photo.shape[:2])
            if longest > VIEW_MAX_DIM:
                factor = VIEW_MAX_DIM / max(1, longest)
                photo = cv2.resize(photo, (int(photo.shape[1] * factor), int(photo.shape[0] * factor)), interpolation=cv2.INTER_AREA)
            return photo
        return create_file_preview(path)
    # Higher-resolution in-app text page for crisper reading when scaled up.
    img = np.full((1980, 1400, 3), (240, 242, 237), np.uint8)
    cv2.rectangle(img, (0, 0), (1400, 132), (42, 78, 98), -1)
    cv2.putText(img, path.name[:58], (36, 84), FONT, 1.15, TECH_WHITE, 2, cv2.LINE_AA)
    lines, y = read_file_text(path).splitlines() or ["(empty file)"], 210
    for line in lines[:56]:
        cv2.putText(img, line.encode("ascii", "replace").decode("ascii")[:96], (40, y), FONT, .78, (34, 40, 45), 1, cv2.LINE_AA)
        y += 31
    cv2.putText(img, "IN-APP READ MODE", (40, 1930), FONT, .7, TECH_BLUE_DARK, 1, cv2.LINE_AA)
    return img


def load_directory(carousel, directory):
    folder = Path(directory); carousel.reset()
    try: entries = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except OSError: return folder
    if folder.parent != folder: carousel.add_file(create_directory_card(folder.parent), str(folder.parent))
    for entry in entries[:14]: carousel.add_file(create_file_preview(entry), str(entry))
    return folder


def load_drives(carousel):
    carousel.reset()
    for drive in list_drives(): carousel.add_file(create_drive_card(drive), str(drive))


# --------------------------------------------------------------------------- #
#  Workspace toolbar (folder / scan / draw / cursor)                          #
# --------------------------------------------------------------------------- #

TOOLBAR_BUTTONS = [
    ("BROWSE", "folder", "OPEN FILES"),
    ("SCAN", "scan", "SCAN NOTEBOOK"),
    ("DRAW", "pencil", "DRAW ON SCREEN"),
    ("MOUSE", "cursor", "CONTROL MOUSE"),
]


def _draw_button_icon(img, kind, rect, color):
    x1, y1, x2, y2 = rect
    pad = int((x2 - x1) * 0.24)
    ix1, iy1, ix2, iy2 = x1 + pad, y1 + pad, x2 - pad, y2 - pad
    if kind == "folder":
        tab_h = max(4, (iy2 - iy1) // 4)
        cv2.rectangle(img, (ix1, iy1 + tab_h), (ix2, iy2), color, 2, cv2.LINE_AA)
        cv2.line(img, (ix1, iy1 + tab_h), (ix1, iy1), color, 2, cv2.LINE_AA)
        cv2.line(img, (ix1, iy1), (ix1 + (ix2 - ix1)//2, iy1), color, 2, cv2.LINE_AA)
        cv2.line(img, (ix1 + (ix2 - ix1)//2, iy1), (ix1 + (ix2 - ix1)//2 + tab_h, iy1 + tab_h), color, 2, cv2.LINE_AA)
    elif kind == "scan":
        cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, 1, cv2.LINE_AA)
        seg = max(4, (ix2 - ix1) // 4)
        for cx, cy, dx, dy in [(ix1, iy1, 1, 1), (ix2, iy1, -1, 1), (ix1, iy2, 1, -1), (ix2, iy2, -1, -1)]:
            cv2.line(img, (cx, cy), (cx + dx*seg, cy), color, 2, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy*seg), color, 2, cv2.LINE_AA)
        my = (iy1 + iy2) // 2
        cv2.line(img, (ix1 + seg//2, my), (ix2 - seg//2, my), TECH_ACCENT, 2, cv2.LINE_AA)
    elif kind == "pencil":
        cv2.line(img, (ix1, iy2), (ix2, iy1), color, 3, cv2.LINE_AA)
        cv2.line(img, (ix1, iy2), (ix1 + 6, iy2 - 6), TECH_WHITE, 2, cv2.LINE_AA)
        cv2.circle(img, (ix2, iy1), 3, TECH_ACCENT, -1, cv2.LINE_AA)
    elif kind == "cursor":
        pts = np.array([[ix1, iy1], [ix1, iy2], [ix1 + (ix2-ix1)//3, iy1 + (iy2-iy1)*2//3],
                        [ix1 + (ix2-ix1)*2//3, iy2 - 2], [ix2, iy1 + (iy2-iy1)//2]], np.int32)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)


def draw_toolbar(img, pointer, config):
    """Vertical left rail. Returns the focused button id, or None."""
    h, w = img.shape[:2]
    btn = max(46, int(66 * config.ui_scale))
    gap = max(12, int(18 * config.ui_scale))
    x1 = 30
    total = len(TOOLBAR_BUTTONS) * (btn + gap) - gap
    y0 = max(80, h // 2 - total // 2)
    focused = None
    for i, (bid, icon, label) in enumerate(TOOLBAR_BUTTONS):
        yb = y0 + i * (btn + gap)
        rect = (x1, yb, x1 + btn, yb + btn)
        inside = bool(pointer and rect[0] <= pointer[0] <= rect[2] and rect[1] <= pointer[1] <= rect[3])
        color = TECH_ACCENT if inside else TECH_BLUE
        rounded_panel(img, (rect[0], rect[1]), (rect[2], rect[3]), (12, 26, 38), .82, color, 2 if inside else 1)
        _draw_button_icon(img, icon, rect, color)
        if inside:
            lw = cv2.getTextSize(label, FONT, .46, 1)[0][0]
            rounded_panel(img, (rect[2] + 8, yb + btn//2 - 16), (rect[2] + 24 + lw, yb + btn//2 + 12), (10, 22, 34), .8, TECH_BLUE_DARK, 1)
            glow_text(img, label, (rect[2] + 16, yb + btn//2 + 6), .46, TECH_WHITE, 1)
            focused = bid
    return focused


class CarouselEngine:
    def __init__(self): self.reset()
    def reset(self):
        self.files, self.file_paths, self.scroll_float, self.offset, self.scale = [], [], 0.0, [0.0, 0.0], 1.0
    def add_file(self, texture, file_path=None):
        self.files.append(texture); self.file_paths.append(file_path)
    def focused_index(self): return int(round(self.scroll_float)) if self.files else -1
    def focused_path(self):
        index = self.focused_index()
        return self.file_paths[index] if 0 <= index < len(self.file_paths) else None
    def label_for(self, index):
        if 0 <= index < len(self.file_paths) and self.file_paths[index]: return Path(self.file_paths[index]).name or str(Path(self.file_paths[index]))
        return None
    def update(self, x, width):
        if x is None or not self.files: return
        target = np.clip((x - .20*width) / max(1, .60*width), 0, 1) * (len(self.files)-1)
        self.scroll_float += (target - self.scroll_float) * .2
    def manipulate(self, midpoint, span, anchor):
        if midpoint is None or span is None: return None
        if anchor is None: return (midpoint, span, list(self.offset), self.scale)
        a_mid, a_span, a_off, a_scale = anchor
        if abs(span - a_span) / max(1.0, a_span) > 0.35: return None
        self.offset = [a_off[0] + (midpoint[0]-a_mid[0]), a_off[1] + (midpoint[1]-a_mid[1])]
        self.scale = float(np.clip(a_scale * span/max(1.0, a_span), 0.55, 2.4))
        return anchor
    def draw(self, img, config):
        h, w = img.shape[:2]
        if not self.files:
            glow_text(img, "// ARCHIVE EMPTY", (w//2, h//2), .7, centered=True); return img

        base_scale = max(0.1, self.scale * config.ui_scale)
        base_h = int(h * .40 * base_scale)
        base_w = int(base_h * .72)
        spacing = int(base_h * 1.02)
        cx0, cy0 = w//2 + int(self.offset[0]), int(h*.50) + int(self.offset[1])
        focus = self.focused_index()
        focused_cx, focused_cy, focused_ph = cx0, cy0, base_h

        for i in sorted(range(len(self.files)), key=lambda n: abs(n-self.scroll_float), reverse=True):
            diff, distance = i-self.scroll_float, abs(i-self.scroll_float)
            scale = max(.45, 1-distance*.22)
            # Enforce minimum dimension sizes to prevent OpenCV Matrix division errors
            pw, ph = max(2, int(base_w*scale)), max(2, int(base_h*scale))
            cx, cy = cx0 + int(diff*spacing), cy0 + int(distance*18)
            pts = [[cx-pw//2, cy-ph//2], [cx+pw//2, cy-ph//2], [cx+pw//2, cy+ph//2], [cx-pw//2, cy+ph//2]]

            img, poly = paste_texture(img, self.files[i], pts, max(.35, 1-distance*.38))
            cv2.polylines(img, [poly], True, TECH_ACCENT if i == focus else TECH_BLUE_DARK, 3 if i == focus else 1, cv2.LINE_AA)
            if i == focus:
                glow_text(img, f"[ {i+1:02d} / {len(self.files):02d} ]", (cx-pw//2, cy-ph//2-14), .6, TECH_WHITE)
                focused_cx, focused_cy, focused_ph = cx, cy, ph

        # Draw dynamic, self-sizing label attached below the focused card
        if name := self.label_for(focus):
            text_scale = max(0.4, 0.82 * base_scale)
            text_thick = max(1, int(2 * base_scale))
            display_name = name[:52]

            text_size = cv2.getTextSize(display_name, FONT, text_scale, text_thick)[0]

            banner_y = focused_cy + focused_ph//2 + int(40 * base_scale)
            pad_x, pad_y = int(20 * base_scale), int(12 * base_scale)

            x1 = focused_cx - text_size[0]//2 - pad_x
            x2 = focused_cx + text_size[0]//2 + pad_x
            y1 = banner_y - text_size[1] - pad_y
            y2 = banner_y + pad_y

            translucent_rect(img, (x1, y1), (x2, y2), (10, 22, 34), .72)
            cv2.rectangle(img, (x1, y1), (x2, y2), TECH_BLUE_DARK, 1)
            glow_text(img, display_name, (focused_cx - text_size[0]//2, banner_y - pad_y//2), text_scale, TECH_WHITE, text_thick, centered=False)

        return img


class HoloPanelEngine:
    def __init__(self): self.center, self.size, self.texture, self.is_active = None, 280, None, False
    def draw(self, img, t, config):
        if not self.is_active or self.texture is None: return img, None
        h, w = img.shape[:2]
        self.size = int(np.clip(self.size, 120, max(120, min(w, h)-40)))
        if self.center is None: self.center = [w//2, h//2]

        # Guard minimum bounds against division errors
        half = max(2, int((self.size//2) * config.ui_scale))
        self.center = [int(np.clip(self.center[0], half, w-half)), int(np.clip(self.center[1], half, h-half))]
        x, y = self.center
        aspect = self.texture.shape[0] / max(1, self.texture.shape[1])
        hh = max(2, int(half * aspect))

        img, poly = paste_texture(img, self.texture, [[x-half,y-hh],[x+half,y-hh],[x+half,y+hh],[x-half,y+hh]])
        cv2.polylines(img, [poly], True, TECH_BLUE_DARK, 6, cv2.LINE_AA); cv2.polylines(img, [poly], True, TECH_BLUE, 2, cv2.LINE_AA)
        glow_text(img, "// ACTIVE DOCUMENT", (x-half, y-hh-12), .5)

        # Dedicated Close Button attached to the workspace panel
        btn_size = max(30, int(40 * config.ui_scale))
        bx, by = x + half - btn_size, y - hh - btn_size - 10
        translucent_rect(img, (bx, by), (bx+btn_size, by+btn_size), (40, 10, 10), 0.8)
        cv2.rectangle(img, (bx, by), (bx+btn_size, by+btn_size), (150, 50, 50), 2)

        margin = max(6, int(12 * config.ui_scale))
        cv2.line(img, (bx+margin, by+margin), (bx+btn_size-margin, by+btn_size-margin), TECH_WHITE, 2, cv2.LINE_AA)
        cv2.line(img, (bx+btn_size-margin, by+margin), (bx+margin, by+btn_size-margin), TECH_WHITE, 2, cv2.LINE_AA)

        return img, (bx, by, bx+btn_size, by+btn_size)


# --------------------------------------------------------------------------- #
#  Notebook scanner (quadrilateral detection -> perspective scan -> PDF)      #
# --------------------------------------------------------------------------- #

def _order_quad(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


class DocumentScanner:
    """Detects a held notebook/page as a quadrilateral and saves a flattened PDF."""

    def __init__(self):
        self.stable_since = None
        self.prev_quad = None
        self.saved_path = None
        self.flash_until = 0.0
        self.cooldown_until = 0.0
        self.status = "HOLD A NOTEBOOK UP TO THE CAMERA"
        self.hold = 1.3

    def reset(self):
        self.stable_since = None
        self.prev_quad = None
        self.status = "HOLD A NOTEBOOK UP TO THE CAMERA"

    def find_quad(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 55, 170)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        best, best_area = None, 0.0
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            area = cv2.contourArea(c)
            if area < 0.09 * frame_area or area > 0.97 * frame_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx) and area > best_area:
                best_area, best = area, approx.reshape(4, 2)
        return best

    def _similar(self, a, b):
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        centroid_shift = np.linalg.norm(ca - cb)
        area_a = cv2.contourArea(a.astype(np.float32))
        area_b = cv2.contourArea(b.astype(np.float32))
        area_ratio = min(area_a, area_b) / max(1.0, max(area_a, area_b))
        return centroid_shift < 45 and area_ratio > 0.82

    def warp(self, frame, quad):
        rect = _order_quad(quad)
        (tl, tr, br, bl) = rect
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        width, height = max(width, 10), max(height, 10)
        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(frame, matrix, (width, height))

    def enhance(self, warped):
        """Clean, high-contrast 'scanned document' look."""
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        try:
            gray = cv2.fastNlMeansDenoising(gray, None, 9, 7, 21)
        except Exception:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        scan = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 21, 12)
        return cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR)

    def _save_pdf(self, image):
        SCAN_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = SCAN_DIR / f"scan_{stamp}.pdf"
        try:
            from PIL import Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).convert("RGB").save(str(pdf_path), "PDF", resolution=200.0)
            return pdf_path
        except Exception:
            # Fallback: at least persist the scan as a PNG so nothing is lost.
            png_path = SCAN_DIR / f"scan_{stamp}.png"
            cv2.imwrite(str(png_path), image)
            return png_path

    def _capture(self, source_frame, quad, t):
        scan = self.enhance(self.warp(source_frame, quad))
        self.saved_path = self._save_pdf(scan)
        self.flash_until = t + 0.55
        self.cooldown_until = t + 2.6
        self.stable_since = None
        self.status = f"SAVED -> {self.saved_path.name}"

    def process(self, detect_frame, source_frame, t):
        """Returns (quad_or_None, progress 0..1). Auto-captures when steady."""
        if t < self.cooldown_until:
            return self.prev_quad, 1.0

        quad = self.find_quad(detect_frame)
        progress = 0.0
        if quad is not None:
            if self.prev_quad is not None and self._similar(quad, self.prev_quad):
                if self.stable_since is None:
                    self.stable_since = t
                progress = min(1.0, (t - self.stable_since) / self.hold)
                if progress >= 1.0:
                    self._capture(source_frame, quad, t)
                    return quad, 1.0
                self.status = "HOLD STEADY..." if progress > 0.15 else "NOTEBOOK DETECTED"
            else:
                self.stable_since = t
                self.status = "NOTEBOOK DETECTED"
            self.prev_quad = quad
        else:
            self.stable_since, self.prev_quad = None, None
            self.status = "HOLD A NOTEBOOK UP TO THE CAMERA"
        return quad, progress

    def draw(self, img, quad, progress, t, config):
        h, w = img.shape[:2]
        if t < self.flash_until:
            translucent_rect(img, (0, 0), (w, h), (255, 255, 255), 0.6)

        if quad is not None:
            poly = _order_quad(quad).astype(np.int32)
            col = TECH_ACCENT if progress > 0.15 else TECH_BLUE
            cv2.polylines(img, [poly], True, col, 3, cv2.LINE_AA)
            for (px, py) in poly:
                cv2.circle(img, (int(px), int(py)), 7, TECH_WHITE, -1, cv2.LINE_AA)
                cv2.circle(img, (int(px), int(py)), 7, col, 2, cv2.LINE_AA)
            cx, cy = poly.mean(axis=0).astype(int)
            if progress > 0:
                cv2.ellipse(img, (cx, cy), (46, 46), -90, 0, int(360 * progress), TECH_ACCENT, 5, cv2.LINE_AA)

        rounded_panel(img, (w//2 - 300, h - 96), (w//2 + 300, h - 58), (10, 22, 34), .78, TECH_BLUE_DARK, 1)
        glow_text(img, self.status, (w//2, h - 70), .58, TECH_WHITE, 2, True)
        glow_text(img, f"SCAN MODULE  |  SAVES TO: {SCAN_DIR}", (w//2, h - 40), .40, TECH_BLUE, 1, True)


# --------------------------------------------------------------------------- #
#  Draw-on-screen canvas                                                       #
# --------------------------------------------------------------------------- #

class DrawCanvas:
    def __init__(self):
        self.strokes = []
        self.current = None

    def update(self, drawing, pt, color=DRAW_COLOR):
        if drawing and pt:
            p = (int(pt[0]), int(pt[1]))
            if self.current is None:
                self.current = [color, [p]]
                self.strokes.append(self.current)
            else:
                self.current[1].append(p)
        else:
            self.current = None

    def render(self, img):
        for color, pts in self.strokes:
            if len(pts) == 1:
                cv2.circle(img, pts[0], 3, color, -1, cv2.LINE_AA)
            for i in range(1, len(pts)):
                cv2.line(img, pts[i-1], pts[i], (0, 0, 0), 6, cv2.LINE_AA)
                cv2.line(img, pts[i-1], pts[i], color, 3, cv2.LINE_AA)

    def clear(self):
        self.strokes, self.current = [], None


# --------------------------------------------------------------------------- #
#  Hand-driven OS mouse control                                                #
# --------------------------------------------------------------------------- #

class MouseController:
    def __init__(self):
        self.available = False
        self.last_click = 0.0
        try:
            import pyautogui
            pyautogui.FAILSAFE = False
            self.pg = pyautogui
            self.sw, self.sh = pyautogui.size()
            self.available = True
        except Exception:
            self.pg = None

    def control(self, pointer, pinch, w, h, t, img):
        hh, ww = img.shape[:2]
        if not self.available:
            glow_text(img, "MOUSE CONTROL NEEDS 'pyautogui'", (ww//2, hh//2), .6, TECH_RED, 2, True)
            glow_text(img, "pip install pyautogui", (ww//2, hh//2 + 34), .5, TECH_WHITE, 1, True)
            return
        if pointer:
            nx = pointer[8][1] / max(1, w)
            ny = pointer[8][2] / max(1, h)
            # Map the comfortable central 70% of the frame to the full screen.
            mx = float(np.clip((nx - 0.15) / 0.70, 0.0, 1.0))
            my = float(np.clip((ny - 0.15) / 0.70, 0.0, 1.0))
            try:
                self.pg.moveTo(int(mx * self.sw), int(my * self.sh), _pause=False)
                if pinch and t - self.last_click > 0.55:
                    self.pg.click()
                    self.last_click = t
            except Exception:
                pass


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
def is_peace_sign(lm):
    if not lm: return False
    scale = hand_scale(lm)
    index_up = lm[8][2] < lm[6][2] - 0.1 * scale
    middle_up = lm[12][2] < lm[10][2] - 0.1 * scale
    ring_down = lm[16][2] > lm[14][2]
    pinky_down = lm[20][2] > lm[18][2]
    v_shape = math.hypot(lm[8][1]-lm[12][1], lm[8][2]-lm[12][2]) > 0.25 * scale
    return index_up and middle_up and ring_down and pinky_down and v_shape
def get_hands(results, w, h):
    hands = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hands[handedness.classification[0].label] = [[i, int(p.x*w), int(p.y*h)] for i,p in enumerate(landmarks.landmark)]
    return hands
def smooth_point(old, new, alpha=.5):
    return list(new) if old is None else [old[0]+(new[0]-old[0])*alpha, old[1]+(new[1]-old[1])*alpha]


class PeaceSignDetector:
    def __init__(self): self.start_time = None
    def update(self, left, right, now):
        if is_peace_sign(left) or is_peace_sign(right):
            if self.start_time is None: self.start_time = now
            elif now - self.start_time > 0.8: return True
        else: self.start_time = None
        return False
    def prompt(self): return "HOLD A PEACE SIGN TO INITIALIZE" if not self.start_time else "INITIALIZING SYSTEM..."


class SelectionTimer:
    """Timer that latches once completed, preventing double-triggers until pinch is released."""
    def __init__(self):
        self.target_id = None
        self.start_time = None
        self.latched = False

    def update(self, is_active, target_id, t, duration):
        if not is_active:
            self.target_id, self.start_time = None, None
            self.latched = False
            return False, 0.0

        if self.latched:
            return False, 0.0

        if target_id is None:
            self.target_id, self.start_time = None, None
            return False, 0.0

        if target_id != self.target_id or self.start_time is None:
            self.target_id, self.start_time = target_id, t
            return False, 0.0

        progress = min(1.0, (t - self.start_time) / max(0.1, duration))
        if progress >= 1.0:
            self.latched = True
            return True, 1.0
        return False, progress


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
    glow_text(img, "POINT TO FOCUS  |  PINCH & HOLD TO SELECT", (cx, int(h*.20)), .40, TECH_WHITE, 1, True)

    cards = [
        ("01", "WORKSPACE", "Browse and manipulate active documents", True),
        ("02", "SYSTEMS", "Reserved for future modules", False),
        ("03", "SETTINGS", "Adjust HUD scale and gesture timing", True)
    ]
    card_w, card_h, gap = min(360, int(w*.55)), 64, 28
    selected = None
    for i, (num, title, subtitle, enabled) in enumerate(cards):
        y1 = cy - card_h - gap + i*(card_h+gap); y2 = y1+card_h; x1, x2 = cx-card_w//2, cx+card_w//2
        inside = pointer and x1 <= pointer[0] <= x2 and y1 <= pointer[1] <= y2
        color = TECH_ACCENT if inside and enabled else (TECH_BLUE if enabled else TECH_BLUE_DARK)
        translucent_rect(img, (x1,y1), (x2,y2), (9,20,30), .72)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2 if inside else 1, cv2.LINE_AA)
        glow_text(img, num, (x1+16, y1+29), .55, color, 2)
        glow_text(img, title, (x1+66, y1+27), .50, TECH_WHITE if enabled else TECH_BLUE_DARK, 2)
        glow_text(img, subtitle, (x1+66, y1+49), .34, color)
        if inside and enabled: selected = title
    return selected


def draw_settings_menu(img, pointer, p_pinch, t, config):
    h, w = img.shape[:2]; cx, cy = w//2, h//2
    translucent_rect(img, (0,0), (w,h), (4,8,12), .7)
    glow_text(img, "SYSTEM SETTINGS", (cx, int(h*.20)), .75, TECH_BLUE, 2, True)
    glow_text(img, "PINCH AND DRAG TO ADJUST VALUES", (cx, int(h*.25)), .40, TECH_WHITE, 1, True)

    sliders = [
        ("UI SCALE MULTIPLIER", "ui_scale", 0.5, 2.0, config.ui_scale),
        ("PINCH HOLD DURATION (SEC)", "pinch_duration", 0.2, 1.5, config.pinch_duration)
    ]

    for i, (label, key, min_v, max_v, val) in enumerate(sliders):
        y = int(h * 0.45) + i * 120
        glow_text(img, f"{label}: {val:.2f}", (cx, y - 25), 0.45, TECH_WHITE, 1, True)
        bar_w = 400
        bx1, bx2 = cx - bar_w//2, cx + bar_w//2
        cv2.rectangle(img, (bx1, y), (bx2, y+8), TECH_BLUE_DARK, 2)

        ratio = (val - min_v) / max(0.1, (max_v - min_v))
        hx = bx1 + int(ratio * bar_w)

        hover = pointer and (y-20 <= pointer[1] <= y+28) and (bx1-20 <= pointer[0] <= bx2+20)
        color = TECH_ACCENT if hover else TECH_BLUE
        cv2.circle(img, (hx, y+4), 12, color, -1)

        if hover and p_pinch:
            new_ratio = np.clip((pointer[0] - bx1) / bar_w, 0.0, 1.0)
            setattr(config, key, min_v + new_ratio * (max_v - min_v))


NAV_TRAIL = {
    "MENU": "MENU",
    "SETTINGS": "MENU / SETTINGS",
    "CAROUSEL": "MENU / WORKSPACE",
    "DRIVE_SELECT": "MENU / WORKSPACE / DEVICE / DRIVE",
    "DEVICE_BROWSER": "MENU / WORKSPACE / DEVICE / DRIVE / FOLDER",
    "WORKSPACE": "MENU / WORKSPACE / DOCUMENT (USE [X] BUTTON TO CLOSE)",
    "SCANNER": "MENU / WORKSPACE / SCAN NOTEBOOK",
    "DRAW": "MENU / WORKSPACE / DRAW",
    "MOUSE_CONTROL": "MENU / WORKSPACE / MOUSE CONTROL",
}


def draw_navigation_hint(img, mode):
    trail = NAV_TRAIL.get(mode, "MENU")
    glow_text(img, trail, (24, 24), .42, TECH_BLUE, 1)
    if mode not in {"WORKSPACE", "MOUSE_CONTROL"}:
        glow_text(img, "HOLD CLOSED FIST: BACK   |   m: MAIN MENU", (img.shape[1]-340, 24), .38, TECH_WHITE, 1)


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
    return cap


def main():
    cap = open_camera()
    if not cap.isOpened(): print("[JARVIS] ERROR: could not open a camera."); return
    mp_hands = mp.solutions.hands
    detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=.7, min_tracking_confidence=.6)

    config = AppConfig()
    workspace, carousel, peace_detector = HoloPanelEngine(), CarouselEngine(), PeaceSignDetector()
    display_filter, device_directory = StudioCameraFilter(), Path.home()
    scanner, draw_canvas, mouse = DocumentScanner(), DrawCanvas(), MouseController()

    sel_timer = SelectionTimer()

    mode, grabbed, reticle_pt, fullscreen = "WELCOME", False, None, False
    resize_anchor, fist_started, fist_latched = None, None, False
    carousel_anchor = None
    mouse_win_small = False

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO); cv2.resizeWindow(WIN, 1280, 720)
    fps, last, frames = 0., time.time(), 0

    def restore_window():
        nonlocal mouse_win_small
        if mouse_win_small:
            cv2.resizeWindow(WIN, 1280, 720)
            mouse_win_small = False

    while True:
        ok, raw = cap.read()
        if not ok: break
        t = time.time(); raw = cv2.flip(raw, 1)
        img = display_filter.apply(raw); h, w = img.shape[:2]
        tracking = low_quality_tracking_frame(raw)
        results = detector.process(cv2.cvtColor(tracking, cv2.COLOR_BGR2RGB))
        hands = get_hands(results, w, h); left, right = hands.get("Left"), hands.get("Right"); pointer = right or left
        p_pinch, p_pt = is_pinching(pointer) if pointer else (False, None)
        l_pinch, l_pt = is_pinching(left); r_pinch, r_pt = is_pinching(right)

        pointer_pt = (pointer[8][1], pointer[8][2]) if pointer else None
        reticle_pt = smooth_point(reticle_pt, pointer_pt) if pointer_pt else None

        both_palms_open = is_palm_open(left) and is_palm_open(right)
        closed_fist = any(is_closed_fist(hand) for hand in hands.values())

        if closed_fist: fist_started = t if fist_started is None else fist_started
        else: fist_started, fist_latched = None, False

        # Fist logic disabled globally inside WORKSPACE (document view).
        back_modes = {"CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER", "SETTINGS", "SCANNER", "DRAW", "MOUSE_CONTROL"}
        back_requested = (mode in back_modes and not fist_latched
                          and fist_started is not None and t - fist_started >= .60)

        if back_requested:
            fist_latched = True
            if mode in {"CAROUSEL", "SETTINGS"}: mode = "MENU"
            elif mode in {"SCANNER", "DRAW"}: mode = "CAROUSEL"
            elif mode == "MOUSE_CONTROL": restore_window(); mode = "CAROUSEL"
            elif mode == "DRIVE_SELECT": mode = "CAROUSEL"
            elif mode == "DEVICE_BROWSER":
                parent = device_directory.parent
                if any(device_directory == d for d in list_drives()) or parent == device_directory:
                    mode = "DRIVE_SELECT"; load_drives(carousel)
                else:
                    device_directory = load_directory(carousel, parent)

        carousel_manip = both_palms_open and mode in {"CAROUSEL", "DRIVE_SELECT", "DEVICE_BROWSER"} and not closed_fist
        if carousel_manip:
            midpoint = ((left[9][1]+right[9][1])//2, (left[9][2]+right[9][2])//2)
            span = math.dist(left[9][1:], right[9][1:])
            carousel_anchor = carousel.manipulate(midpoint, span, carousel_anchor)
        else:
            carousel_anchor = None

        def scroll_carousel():
            if not carousel_manip: carousel.update(pointer[8][1] if pointer else None, w)

        progress_arc = 0.0

        if mode == "WELCOME":
            triggered = peace_detector.update(left, right, t)
            draw_welcome(img, t, peace_detector.prompt())
            if triggered: mode = "MENU"

        elif mode == "MENU":
            choice = draw_main_menu(img, reticle_pt, t)
            draw_navigation_hint(img, "MENU")
            triggered, progress_arc = sel_timer.update(p_pinch, choice, t, config.pinch_duration)
            if triggered and choice == "WORKSPACE": mode = "CAROUSEL"
            if triggered and choice == "SETTINGS": mode = "SETTINGS"

        elif mode == "SETTINGS":
            draw_navigation_hint(img, "SETTINGS")
            draw_settings_menu(img, pointer_pt, p_pinch, t, config)

        else:
            translucent_rect(img, (0,0), (w,54), HUD_BG, .55); translucent_rect(img, (0,h-38), (w,h), HUD_BG, .55); corner_brackets(img)
            if carousel_manip:
                glow_text(img, f"MOVE + SCALE ARCHIVE  x{carousel.scale:0.2f}", (w//2, 48), .5, TECH_ACCENT, 1, True)

            if mode == "CAROUSEL":
                draw_navigation_hint(img, "CAROUSEL")
                glow_text(img, "PICK A TOOL FROM THE LEFT RAIL   |   BOTH PALMS: MOVE / RESIZE", (24,48), .34, TECH_WHITE)
                scroll_carousel(); img = carousel.draw(img, config)

                focus_btn = draw_toolbar(img, reticle_pt, config)
                triggered, progress_arc = sel_timer.update(p_pinch and not carousel_manip, focus_btn, t, config.pinch_duration)
                if triggered:
                    if focus_btn == "BROWSE":
                        load_drives(carousel); mode = "DRIVE_SELECT"
                    elif focus_btn == "SCAN":
                        scanner.reset(); mode = "SCANNER"
                    elif focus_btn == "DRAW":
                        mode = "DRAW"
                    elif focus_btn == "MOUSE":
                        mode = "MOUSE_CONTROL"

            elif mode == "DRIVE_SELECT":
                draw_navigation_hint(img, "DRIVE_SELECT")
                glow_text(img, "STEP 1 OF 3: PICK A DRIVE   |   ONE-HAND PINCH & HOLD: OPEN", (24,48), .34, TECH_WHITE)
                scroll_carousel(); img = carousel.draw(img, config)

                focused_path = carousel.focused_path()
                triggered, progress_arc = sel_timer.update(p_pinch and not carousel_manip, focused_path, t, config.pinch_duration)
                if triggered:
                    device_directory = load_directory(carousel, focused_path)
                    mode = "DEVICE_BROWSER"

            elif mode == "DEVICE_BROWSER":
                draw_navigation_hint(img, "DEVICE_BROWSER")
                glow_text(img, f"STEP 2-3: {str(device_directory)[:60]}  |  PINCH & HOLD: OPEN FOLDER / FILE", (24,48), .34, TECH_WHITE)
                scroll_carousel(); img = carousel.draw(img, config)

                focused_path = carousel.focused_path()
                is_dir = Path(focused_path).is_dir() if focused_path else False
                is_file = Path(focused_path).is_file() if focused_path else False

                triggered, progress_arc = sel_timer.update(p_pinch and not carousel_manip, focused_path, t, config.pinch_duration)

                if triggered:
                    if is_dir:
                        device_directory = load_directory(carousel, focused_path)
                    elif is_file:
                        workspace.texture = create_document_texture(focused_path)
                        workspace.is_active = True
                        workspace.center, workspace.size = [w//2, h//2], int(min(w, h)*.7)
                        mode = "WORKSPACE"

            elif mode == "SCANNER":
                draw_navigation_hint(img, "SCANNER")
                glow_text(img, "HOLD A NOTEBOOK / PAGE FLAT TO THE CAMERA   |   AUTO-SAVES AS PDF   |   HOLD FIST: BACK", (24,48), .32, TECH_WHITE)
                quad, scan_prog = scanner.process(raw, raw, t)
                scanner.draw(img, quad, scan_prog, t, config)

            elif mode == "DRAW":
                draw_navigation_hint(img, "DRAW")
                glow_text(img, "PINCH TO DRAW   |   c: CLEAR   |   HOLD FIST: BACK", (24,48), .34, TECH_WHITE)
                draw_canvas.update(bool(p_pinch), pointer_pt)
                draw_canvas.render(img)
                if pointer_pt:
                    cv2.circle(img, (int(pointer_pt[0]), int(pointer_pt[1])), 6, DRAW_COLOR, -1, cv2.LINE_AA)

            elif mode == "MOUSE_CONTROL":
                if not mouse_win_small:
                    cv2.resizeWindow(WIN, 400, 240)
                    try: cv2.moveWindow(WIN, 40, 40)
                    except Exception: pass
                    mouse_win_small = True
                draw_navigation_hint(img, "MOUSE_CONTROL")
                glow_text(img, "MOVE HAND: CURSOR  |  PINCH: CLICK  |  FIST: EXIT", (24,48), .34, TECH_WHITE)
                mouse.control(pointer, bool(p_pinch), w, h, t, img)

            elif mode == "WORKSPACE":
                draw_navigation_hint(img, "WORKSPACE")
                glow_text(img, "PINCH ON DOCUMENT: MOVE   |   TWO-HAND PINCH: RESIZE   |   PINCH 'X': CLOSE", (24,48), .34, TECH_WHITE)

                # Render document and its attached close button first
                img, close_rect = workspace.draw(img, t, config)

                # Check interaction specifically for the UI close button
                hover_close = False
                if close_rect and pointer_pt:
                    bx1, by1, bx2, by2 = close_rect
                    hover_close = bx1 <= pointer_pt[0] <= bx2 and by1 <= pointer_pt[1] <= by2

                triggered, progress_arc = sel_timer.update(p_pinch and hover_close, "CLOSE_DOC", t, config.pinch_duration)

                if triggered:
                    # Return to the exact Device Browser context to resume exploration
                    mode, workspace.is_active, grabbed = "DEVICE_BROWSER", False, False

                elif not hover_close:
                    if l_pinch and r_pinch and l_pt and r_pt:
                        span = max(1.0, math.dist(l_pt, r_pt))
                        if resize_anchor is None: resize_anchor = (span, workspace.size)
                        start_span, start_size = resize_anchor
                        workspace.size = int(np.clip(start_size * span / max(1.0, start_span), 100, min(w,h)-40)); grabbed = False
                        cv2.line(img, l_pt, r_pt, TECH_ACCENT, 1, cv2.LINE_AA)
                        glow_text(img, f"RESIZE {workspace.size}px", ((l_pt[0]+r_pt[0])//2, (l_pt[1]+r_pt[1])//2-12), .38, TECH_ACCENT, 1, True)
                    elif p_pinch and p_pt:
                        resize_anchor = None
                        cx, cy, half = workspace.center[0], workspace.center[1], int((workspace.size//2) * config.ui_scale)
                        # Only grab if pinching inside the actual document space
                        if cx-half < p_pt[0] < cx+half and cy-half < p_pt[1] < cy+half:
                            grabbed = True
                        if grabbed: workspace.center = list(p_pt)
                    else:
                        grabbed, resize_anchor = False, None

            if mode != "MOUSE_CONTROL":
                reticle(img, reticle_pt, t, progress_arc)

        frames += 1
        if t-last >= .5: fps, frames, last = frames/max(0.001, (t-last)), 0, t
        glow_text(img, f"VISION: {len(hands)} HANDS | {fps:4.1f} FPS | DISPLAY: STUDIO DENOISE | AI: RAW {tracking.shape[1]}x{tracking.shape[0]}", (20,h-14), .36, TECH_BLUE)
        glow_text(img, "m: MENU   f: FULLSCREEN   q: QUIT", (w-240,h-14), .36, TECH_BLUE_DARK)

        cv2.imshow(WIN, img)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key == ord('c') and mode == "DRAW":
            draw_canvas.clear()
        if key == ord('m') and mode != "WELCOME":
            restore_window()
            mode, workspace.is_active, grabbed = "MENU", False, False
        if key == ord('f'):
            fullscreen = not fullscreen; cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1: break

    cap.release(); detector.close(); cv2.destroyAllWindows()


if __name__ == "__main__": main()
