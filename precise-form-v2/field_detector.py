"""
field_detector.py

Stage 1 of the Form Recognizer pipeline.

Uses OpenCV morphological operations and contour analysis to detect blank
fillable form fields (underlines, boxes, checkboxes) in a page image.

All output bounding boxes are normalized to [0.0, 1.0] relative to
the page image dimensions (x1, y1, x2, y2).

Detection strategy:
  1. Underline fields  — thin horizontal line segments
  2. Checkboxes        — dedicated pass: small, near-square, hollow contours
  3. Box/rectangle fields — larger closed rectangular contours
  4. Fallback           — Canny edge + contour pass for fields missed above

Confidence:
  "high" — detected by primary morphological method, no overlap issues
  "low"  — detected by fallback method, or has atypical geometry, or overlaps
            another detection closely
"""

from __future__ import annotations

import re
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_engine import OCREngine


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedField:
    """A single detected form field on one page."""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) normalized [0,1]
    field_type: str                           # "underline" | "box" | "checkbox" | "slash_option"
    confidence: str                           # "high" | "low"
    # Raw pixel bbox before normalization — useful for debugging
    bbox_px: tuple[int, int, int, int] = field(default=(0, 0, 0, 0), repr=False)
    # For slash_option: list of individual option strings e.g. ["是", "否"]
    options: list[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Page-fraction thresholds for field width (relative to image width)
MIN_FIELD_WIDTH_FRAC = 0.03   # fields must be wider than 3% of page
MAX_FIELD_WIDTH_FRAC = 0.95   # exclude full-width page borders

# Absolute pixel height bounds for underline lines
UNDERLINE_MIN_H_PX = 1
UNDERLINE_MAX_H_PX = 8

# Box detection: area as fraction of total page area
MIN_BOX_AREA_FRAC  = 0.0010  # ~0.10% — raised to reduce noise hits (was 0.05%)
MAX_BOX_AREA_FRAC  = 0.25    # max 25% — exclude whole-page content blocks

# Checkbox-specific size bounds
# At 300 DPI: 1mm ≈ 12px.  Real form checkboxes measured on FORM_WR2: ~49-55px (≈4mm).
CHECKBOX_MIN_SIDE_PX   = 40    # ≈ 3.3 mm — eliminates text chars & noise
CHECKBOX_MAX_SIDE_FRAC = 0.06  # at most 6% of page width per side (was 8%)
CHECKBOX_MIN_FILL      = 0.05  # at least 5% dark pixels  (visible border)
CHECKBOX_MAX_FILL      = 0.40  # at most 40% dark pixels  (was 45%)
CHECKBOX_SQUARENESS    = 0.80  # aspect ratio must be in [0.80, 1.25] (was 0.75)

# Page border exclusion margin (fraction of page dimension)
BORDER_MARGIN_FRAC = 0.02

# Overlap proximity for low-confidence flagging (pixels)
OVERLAP_PROXIMITY_PX = 5

# Morphological kernel lengths for line detection (fraction of page width)
H_LINE_KERNEL_FRAC = 0.05  # horizontal kernel spans at least 5% of page width


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_fields(image: np.ndarray) -> list[DetectedField]:
    """
    Detect all blank fillable form fields in a single page image.

    Args:
        image : BGR numpy array (H, W, 3) — output of pdf_to_image.py

    Returns:
        List of DetectedField, sorted top-to-bottom then left-to-right.

    Field types detected
    --------------------
    underline — thin horizontal line fields (OpenCV morphological)
    box       — rectangular box fields (OpenCV contour)
    checkbox  — small square checkbox (OpenCV contour)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary threshold — dark lines on white background
    # THRESH_BINARY_INV: ink lines → 255 (white), paper → 0 (black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Calibrate character height from this document's actual text.
    char_height_px = _estimate_char_height(binary, w, h)

    fields: list[DetectedField] = []
    fields += _detect_underlines(binary, w, h)
    fields += _detect_checkboxes(binary, w, h)   # dedicated pass BEFORE boxes
    fields += _detect_boxes(binary, w, h)

    # Canny fallback for anything not caught above
    fallback = _detect_fallback_canny(gray, w, h, existing=fields)
    fields += fallback

    # Deduplicate and flag overlaps
    fields = _deduplicate(fields, w, h)

    # Expand underline bboxes upward through blank space until content is hit.
    fields = _expand_underline_heights(fields, binary, w, h, char_height_px)

    # Sort: top-to-bottom, left-to-right
    fields.sort(key=lambda f: (f.bbox[1], f.bbox[0]))

    return fields


# ---------------------------------------------------------------------------
# Document calibration
# ---------------------------------------------------------------------------

def _estimate_char_height(binary: np.ndarray, w: int, h: int) -> int:
    """
    Estimate the typical printed text character height (in pixels) from the
    connected components present in the binary image.

    This gives a document-specific calibration value so that field expansion
    is proportional to the actual font size — regardless of DPI or zoom.

    Strategy:
      - Find all connected components (CCs).
      - Keep only those whose size matches a single text character:
          height in [0.5%, 3%] of page height
          width  in [0.1%, 8%] of page width
          aspect ratio (w/h) in [0.2, 4.0]
      - Return the median height of those CCs.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    min_ch = max(4, int(h * 0.005))   # ≥ 0.5 % of page height
    max_ch = int(h * 0.03)            # ≤ 3 %  (one cap-height, not a full line)
    min_cw = max(2, int(w * 0.001))   # ≥ 0.1 % of page width
    max_cw = int(w * 0.08)            # ≤ 8 %  (not a long line segment)

    heights = []
    for i in range(1, num_labels):    # label 0 = background
        cw  = stats[i, cv2.CC_STAT_WIDTH]
        ch  = stats[i, cv2.CC_STAT_HEIGHT]
        if not (min_ch <= ch <= max_ch):
            continue
        if not (min_cw <= cw <= max_cw):
            continue
        aspect = cw / ch
        if not (0.2 <= aspect <= 4.0):
            continue
        heights.append(ch)

    if len(heights) < 10:
        return max(8, int(h * 0.018))   # safe fallback ≈ 1.8% of page

    return int(np.median(heights))


# ---------------------------------------------------------------------------
# Detection methods
# ---------------------------------------------------------------------------

def _detect_underlines(binary: np.ndarray, w: int, h: int) -> list[DetectedField]:
    """Detect horizontal underline-style fields using a morphological horizontal kernel."""
    min_len_px = int(MIN_FIELD_WIDTH_FRAC * w)
    kernel_len = max(min_len_px, int(H_LINE_KERNEL_FRAC * w))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    # Dilate slightly to merge broken underlines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_lines = cv2.dilate(h_lines, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fields = []
    border_margin_x = int(BORDER_MARGIN_FRAC * w)
    border_margin_y = int(BORDER_MARGIN_FRAC * h)

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Height filter: underlines are thin
        if not (UNDERLINE_MIN_H_PX <= ch <= UNDERLINE_MAX_H_PX):
            continue
        # Width filter
        if not (MIN_FIELD_WIDTH_FRAC * w <= cw <= MAX_FIELD_WIDTH_FRAC * w):
            continue
        # Border exclusion
        if (x < border_margin_x or y < border_margin_y
                or x + cw > w - border_margin_x
                or y + ch > h - border_margin_y):
            continue

        bbox_norm = _normalize(x, y, x + cw, y + ch, w, h)
        fields.append(DetectedField(
            bbox=bbox_norm,
            field_type="underline",
            confidence="high",
            bbox_px=(x, y, x + cw, y + ch),
        ))

    return fields


def _detect_checkboxes(binary: np.ndarray, w: int, h: int) -> list[DetectedField]:
    """
    Dedicated pass for checkbox detection.

    Key differences from _detect_boxes:
    - Very light morphological closing (2x2, 1 iter) to avoid filling the interior
    - Much smaller area threshold (min side 15 px, not area fraction)
    - Allows square aspect ratios (ch ≈ cw is expected for checkboxes)
    - Hollow-interior check via fill ratio [CHECKBOX_MIN_FILL, CHECKBOX_MAX_FILL]

    Two bugs in _detect_boxes prevent checkbox detection:
      1. `if ch >= cw: continue`  — squares have ch == cw, so all are filtered
      2. MIN_BOX_AREA_FRAC is too large for typical small checkboxes
    This dedicated function avoids both issues.
    """
    # Very light closing to bridge tiny line gaps without flooding the interior
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # RETR_LIST — all contours, no hierarchy (avoids missing inner contours)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    border_margin_x = int(BORDER_MARGIN_FRAC * w)
    border_margin_y = int(BORDER_MARGIN_FRAC * h)
    max_side_px = int(CHECKBOX_MAX_SIDE_FRAC * w)

    fields = []
    # Track seen positions to suppress duplicate inner/outer contours for the same box
    seen_positions: set[tuple[int, int]] = set()

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # --- Size: must be small and square-ish ---
        if cw < CHECKBOX_MIN_SIDE_PX or ch < CHECKBOX_MIN_SIDE_PX:
            continue
        if cw > max_side_px or ch > max_side_px:
            continue

        # --- Squareness ---
        aspect = cw / max(ch, 1)
        if not (CHECKBOX_SQUARENESS <= aspect <= 1.0 / CHECKBOX_SQUARENESS):
            continue

        # --- Border exclusion ---
        if (x < border_margin_x or y < border_margin_y
                or x + cw > w - border_margin_x
                or y + ch > h - border_margin_y):
            continue

        # --- Hollow-interior check ---
        # In THRESH_BINARY_INV: ink = 255, paper = 0
        # A checkbox outline: dark border + white interior → low fill ratio
        roi = binary[y:y + ch, x:x + cw]
        fill_ratio = np.count_nonzero(roi) / (cw * ch + 1e-6)
        if not (CHECKBOX_MIN_FILL <= fill_ratio <= CHECKBOX_MAX_FILL):
            continue

        # --- Shape approximation: must be roughly rectangular (4–8 corners) ---
        peri = cv2.arcLength(cnt, True)
        if peri < 1:
            continue
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        if len(approx) < 4 or len(approx) > 8:
            continue

        # --- Interior emptiness: inner ~60% of the bbox must be nearly empty ---
        # This rejects solid/filled blobs and character outlines (O, 0, etc.)
        margin = max(2, min(cw, ch) // 6)
        interior = roi[margin: ch - margin, margin: cw - margin]
        if interior.size > 0:
            interior_fill = np.count_nonzero(interior) / (interior.size + 1e-6)
            if interior_fill > 0.15:
                continue

        # --- Suppress near-duplicate contours (inner vs outer of same box) ---
        # Quantise position to 6px grid
        pos_key = (x // 6, y // 6)
        if pos_key in seen_positions:
            continue
        seen_positions.add(pos_key)

        bbox_norm = _normalize(x, y, x + cw, y + ch, w, h)
        fields.append(DetectedField(
            bbox=bbox_norm,
            field_type="checkbox",
            confidence="high",
            bbox_px=(x, y, x + cw, y + ch),
        ))

    return fields


def _detect_boxes(binary: np.ndarray, w: int, h: int) -> list[DetectedField]:
    """
    Detect larger box-style input fields (rectangular outlines, wider than tall).

    NOTE: Checkboxes (small, square) are intentionally handled by
    _detect_checkboxes() — this function targets larger input regions.

    Precision improvements over earlier versions:
    - RETR_EXTERNAL: only outermost contours (eliminates inner contours of
      tables/figures that are not form fields)
    - 4–6 polygon corners (was 4–8): rejects complex non-rectangular shapes
    - Overall fill ≤ 0.35 (was 0.50): boxes must be substantially hollow
    - Interior-zone fill ≤ 0.12: inner 70% of the bbox must be nearly empty,
      which rejects content-rich regions (text blocks, figures) that happen
      to be surrounded by a border
    - Minimum box height (0.8% of page height): filters out very thin slivers
    """
    # Close small gaps to connect broken rectangle borders
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # RETR_EXTERNAL — only outermost contours; avoids duplicating inner walls
    # of complex nested shapes (table cells, figure borders, etc.)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fields = []
    border_margin_x = int(BORDER_MARGIN_FRAC * w)
    border_margin_y = int(BORDER_MARGIN_FRAC * h)
    page_area = w * h
    checkbox_max_side = int(CHECKBOX_MAX_SIDE_FRAC * w)
    min_box_h_px = max(12, int(h * 0.008))   # ≥ 0.8% page height ≈ 28 px on A4@300 dpi

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # Skip checkbox-sized shapes — handled by _detect_checkboxes
        if cw <= checkbox_max_side and ch <= checkbox_max_side:
            aspect = cw / max(ch, 1)
            if CHECKBOX_SQUARENESS <= aspect <= 1.0 / CHECKBOX_SQUARENESS:
                continue

        # Minimum height: slivers are not form input boxes
        if ch < min_box_h_px:
            continue

        # Area filter
        if not (MIN_BOX_AREA_FRAC * page_area <= area <= MAX_BOX_AREA_FRAC * page_area):
            continue
        # Width filter
        if not (MIN_FIELD_WIDTH_FRAC * w <= cw <= MAX_FIELD_WIDTH_FRAC * w):
            continue
        # Border exclusion
        if (x < border_margin_x or y < border_margin_y
                or x + cw > w - border_margin_x
                or y + ch > h - border_margin_y):
            continue
        # Exclude clearly vertical bars (height > 120% of width)
        if ch > cw * 1.2:
            continue

        # Polygon approximation: clean rectangle has 4–6 corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        roi = binary[y:y + ch, x:x + cw]

        # Overall fill: must be substantially hollow (≤ 35%)
        total_fill = np.count_nonzero(roi) / (area + 1e-6)
        if total_fill > 0.35:
            continue

        # Interior-zone fill: inner ~70% of the bbox must be nearly empty.
        # This rejects content regions (text blocks, figures) surrounded by a
        # border that would otherwise pass the overall-fill check.
        mx = max(2, cw // 7)
        my = max(2, ch // 7)
        interior = roi[my: ch - my, mx: cw - mx]
        if interior.size > 0:
            interior_fill = np.count_nonzero(interior) / (interior.size + 1e-6)
            if interior_fill > 0.12:
                continue

        bbox_norm = _normalize(x, y, x + cw, y + ch, w, h)
        fields.append(DetectedField(
            bbox=bbox_norm,
            field_type="box",
            confidence="high",
            bbox_px=(x, y, x + cw, y + ch),
        ))

    return fields


def _expand_underline_heights(
    fields: list[DetectedField],
    binary: np.ndarray,
    w: int,
    h: int,
    char_height_px: int | None = None,
) -> list[DetectedField]:
    """
    Post-processing: expand every underline bbox upward to cover the answer
    area above the line.

    Uses char_height_px (calibrated from the document's own text) to set
    expansion targets proportional to the actual font size.

    Rules:
      1. Search window = 6× char_height above the underline.
      2. Hard ceiling = nearest overlapping bbox above (never cross another field).
      3. If filled-in text is found within 1.5× char_height of the line:
         expand to encompass it (top = text_top − small padding).
      4. Otherwise (blank form or only distant label text): walk upward through
         blank rows and stop just below the first non-blank row encountered.
         If everything is blank up to the ceiling, extend all the way to it.
      5. Blank threshold = 2% of field width (more tolerant than before).
    """
    if char_height_px is None:
        char_height_px = max(8, int(h * 0.018))

    result: list[DetectedField] = []
    for i, f in enumerate(fields):
        if f.field_type == "underline":
            other_px = [other.bbox_px for j, other in enumerate(fields) if j != i]
            result.append(_expand_single_underline(
                f, binary, w, h, other_px, char_height_px
            ))
        else:
            result.append(f)   # slash_option / box / checkbox: keep as-is
    return result


def _expand_single_underline(
    f: DetectedField,
    binary: np.ndarray,
    w: int,
    h: int,
    other_bboxes_px: list[tuple] | None = None,
    char_height_px: int = 30,
) -> DetectedField:
    x1, y_top_line, x2, y_bot_line = f.bbox_px

    # Search window: 6× char height above the underline.
    search_window = max(char_height_px * 6, int(h * 0.06))

    # ── Hard ceiling: nearest horizontally-overlapping bbox edge above us ──
    ceil_y = max(0, y_top_line - search_window)
    if other_bboxes_px:
        for ox1, oy1, ox2, oy2 in other_bboxes_px:
            if min(x2, ox2) - max(x1, ox1) <= 0:
                continue
            if oy2 <= y_top_line:
                ceil_y = max(ceil_y, oy2)   # push ceiling downward (closer to line)

    search_h = y_top_line - ceil_y
    if search_h < 2:
        new_y1 = max(ceil_y, y_top_line - char_height_px)
        return _make_expanded(f, x1, new_y1, x2, y_bot_line, w, h)

    # ── Scan rows above the underline for content ─────────────────────────
    roi = binary[ceil_y:y_top_line, x1:x2]
    if roi.shape[0] < 1 or roi.shape[1] < 1:
        new_y1 = ceil_y
        return _make_expanded(f, x1, new_y1, x2, y_bot_line, w, h)

    row_counts = np.sum(roi > 0, axis=1)
    # A row is "non-blank" if it has > 2% dark pixels across the field width
    blank_thresh = max(3, int(0.02 * (x2 - x1)))
    text_row_indices = np.where(row_counts > blank_thresh)[0]

    if len(text_row_indices) > 0:
        # Check if the bottom-most cluster is close enough to be a filled answer.
        gap_threshold = max(3, char_height_px // 2)
        gaps = np.where(np.diff(text_row_indices) > gap_threshold)[0]
        cluster = text_row_indices[gaps[-1] + 1:] if len(gaps) > 0 else text_row_indices

        cluster_top_abs = ceil_y + int(cluster[0])
        cluster_bot_abs = ceil_y + int(cluster[-1])
        cluster_h       = max(1, cluster_bot_abs - cluster_top_abs)
        dist_from_line  = y_top_line - cluster_bot_abs

        if dist_from_line <= int(char_height_px * 1.5):
            # ── Filled form: answer text right above the underline ────────
            padding = max(2, int(cluster_h * 0.15))
            new_y1  = max(ceil_y, cluster_top_abs - padding)
            if y_bot_line - new_y1 < char_height_px:
                new_y1 = max(ceil_y, y_bot_line - char_height_px)
            return _make_expanded(f, x1, new_y1, x2, y_bot_line, w, h)

    # ── Blank form (or only far/label text): walk upward through blank rows,
    # stop just below the first non-blank row (text/line/image). ──────────
    new_y1 = ceil_y  # default: all blank → extend to hard ceiling
    for row_idx in range(search_h - 1, -1, -1):
        if row_counts[row_idx] > blank_thresh:
            new_y1 = ceil_y + row_idx + 1  # stop just below this non-blank row
            break

    return _make_expanded(f, x1, new_y1, x2, y_bot_line, w, h)


def _make_expanded(
    f: DetectedField,
    x1: int, new_y1: int, x2: int, y_bot_line: int,
    w: int, h: int,
) -> DetectedField:
    return DetectedField(
        bbox=_normalize(x1, new_y1, x2, y_bot_line, w, h),
        field_type=f.field_type,
        confidence=f.confidence,
        bbox_px=(x1, new_y1, x2, y_bot_line),
    )


def _detect_slash_options_cv(
    binary: np.ndarray,
    w: int,
    h: int,
    char_height_px: int,
) -> list[DetectedField]:
    """
    Detect slash-separated option fields by finding "/" characters directly
    in the binary image using connected-component analysis.

    Strategy (maximum recall — no OCR required)
    -------------------------------------------
    1. Find every connected component whose shape matches a "/" stroke:
         - Height in [0.4×, 1.8×] char_height
         - Width < 75% of height  (slashes are narrow)
         - Pixel correlation: as y increases, x decreases (negative corr)
           — this distinguishes "/" from "|", "l", "1", etc.
    2. For each "/" found, collect all character-sized components in the
       same horizontal band (±0.7× char_height vertically, ±5× char_height
       horizontally) to form an option-group bounding box.
    3. Merge overlapping option-group bboxes (handles multiple slashes in
       one group, e.g. "香港/九龍/新界" has two "/" characters).
    4. Return one DetectedField per merged group, confidence="low".
       A downstream LLM decides whether each bbox is a real slash option
       and extracts the individual choice labels.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    border_margin_x = int(BORDER_MARGIN_FRAC * w)
    border_margin_y = int(BORDER_MARGIN_FRAC * h)

    min_ch = int(char_height_px * 0.4)
    max_ch = int(char_height_px * 1.8)

    # ── Step 1: find "/" candidates ────────────────────────────────────────
    slash_centers: list[tuple[int, int]] = []   # (cx, cy)

    for i in range(1, num_labels):
        x   = stats[i, cv2.CC_STAT_LEFT]
        y   = stats[i, cv2.CC_STAT_TOP]
        cw  = stats[i, cv2.CC_STAT_WIDTH]
        ch  = stats[i, cv2.CC_STAT_HEIGHT]
        cx  = x + cw // 2
        cy  = y + ch // 2

        # Size: slash-height ≈ char height; slash-width << height
        if not (min_ch <= ch <= max_ch):
            continue
        if cw < 2 or cw > ch * 0.75:
            continue
        if stats[i, cv2.CC_STAT_AREA] < 8:
            continue

        # Border exclusion
        if (cx < border_margin_x or cx > w - border_margin_x
                or cy < border_margin_y or cy > h - border_margin_y):
            continue

        # Shape check: "/" has negative (y, x) correlation
        ys, xs = np.where(labels == i)
        if len(ys) < 5:
            continue
        if np.std(xs) < 1.0 or np.std(ys) < 1.0:
            continue
        corr = float(np.corrcoef(ys.astype(float), xs.astype(float))[0, 1])
        if corr > -0.40:        # not diagonal-enough or wrong direction
            continue

        slash_centers.append((cx, cy))

    if not slash_centers:
        return []

    # ── Step 2: build character-component lookup (once, reused per slash) ──
    char_comps: list[tuple[int, int, int, int, int, int]] = []
    # (x1, y1, x2, y2, cx, cy)
    for i in range(1, num_labels):
        x   = stats[i, cv2.CC_STAT_LEFT]
        y   = stats[i, cv2.CC_STAT_TOP]
        cw  = stats[i, cv2.CC_STAT_WIDTH]
        ch  = stats[i, cv2.CC_STAT_HEIGHT]
        if not (int(char_height_px * 0.25) <= ch <= max_ch):
            continue
        if cw < 1:
            continue
        char_comps.append((x, y, x + cw, y + ch, x + cw // 2, y + ch // 2))

    # ── Step 3: for each "/" form an option-group bbox ─────────────────────
    band_h      = int(char_height_px * 0.7)   # vertical tolerance
    max_h_reach = int(char_height_px * 5)     # max horizontal reach from "/"
    pad         = max(2, int(char_height_px * 0.15))

    raw_bboxes: list[tuple[int, int, int, int]] = []

    for (scx, scy) in slash_centers:
        gx1 = scx;  gy1 = scy
        gx2 = scx;  gy2 = scy

        for (cx1, cy1, cx2, cy2, ccx, ccy) in char_comps:
            if abs(ccy - scy) > band_h:
                continue
            if abs(ccx - scx) > max_h_reach:
                continue
            gx1 = min(gx1, cx1);  gy1 = min(gy1, cy1)
            gx2 = max(gx2, cx2);  gy2 = max(gy2, cy2)

        raw_bboxes.append((
            max(0,      gx1 - pad),
            max(0,      gy1 - pad),
            min(w - 1,  gx2 + pad),
            min(h - 1,  gy2 + pad),
        ))

    # ── Step 4: merge overlapping / adjacent bboxes ────────────────────────
    merged = _merge_bboxes_horizontal(raw_bboxes)

    # ── Step 5: convert to DetectedField ───────────────────────────────────
    fields: list[DetectedField] = []
    min_width = int(char_height_px * 0.8)   # at least ~1 char wide

    for (x1, y1, x2, y2) in merged:
        if x2 - x1 < min_width:
            continue
        if (x1 < border_margin_x or y1 < border_margin_y
                or x2 > w - border_margin_x
                or y2 > h - border_margin_y):
            continue

        fields.append(DetectedField(
            bbox       = _normalize(x1, y1, x2, y2, w, h),
            field_type = "slash_option",
            confidence = "low",   # LLM verifies downstream
            bbox_px    = (x1, y1, x2, y2),
            options    = [],      # LLM extracts option text downstream
        ))

    return fields


def _merge_bboxes_horizontal(
    bboxes: list[tuple[int, int, int, int]],
    gap: int = 10,
) -> list[tuple[int, int, int, int]]:
    """
    Merge bboxes that overlap or are within `gap` pixels of each other
    horizontally (and share vertical extent).  Returns merged list.
    """
    if not bboxes:
        return []

    # Sort left-to-right
    sorted_boxes = sorted(bboxes, key=lambda b: b[0])
    merged = [list(sorted_boxes[0])]

    for x1, y1, x2, y2 in sorted_boxes[1:]:
        last = merged[-1]
        # Horizontally close/overlapping AND vertically overlapping
        h_close  = x1 <= last[2] + gap
        v_overlap = y1 <= last[3] and y2 >= last[1]
        if h_close and v_overlap:
            last[0] = min(last[0], x1)
            last[1] = min(last[1], y1)
            last[2] = max(last[2], x2)
            last[3] = max(last[3], y2)
        else:
            merged.append([x1, y1, x2, y2])

    return [tuple(b) for b in merged]


def _detect_fallback_canny(
    gray: np.ndarray,
    w: int,
    h: int,
    existing: list[DetectedField],
) -> list[DetectedField]:
    """
    Fallback pass using Canny edges to catch fields missed by morphological methods.
    All detections here are marked low confidence.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    existing_px = [f.bbox_px for f in existing]

    fields = []
    border_margin_x = int(BORDER_MARGIN_FRAC * w)
    border_margin_y = int(BORDER_MARGIN_FRAC * h)
    page_area = w * h

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        if not (MIN_BOX_AREA_FRAC * page_area <= area <= MAX_BOX_AREA_FRAC * page_area):
            continue
        if not (MIN_FIELD_WIDTH_FRAC * w <= cw <= MAX_FIELD_WIDTH_FRAC * w):
            continue
        if ch > cw * 1.2:
            continue
        if (x < border_margin_x or y < border_margin_y
                or x + cw > w - border_margin_x
                or y + ch > h - border_margin_y):
            continue

        # Skip if substantially overlapping an already-detected field
        if _overlaps_any(x, y, x + cw, y + ch, existing_px, threshold=0.5):
            continue

        bbox_norm = _normalize(x, y, x + cw, y + ch, w, h)
        fields.append(DetectedField(
            bbox=bbox_norm,
            field_type="box",
            confidence="low",
            bbox_px=(x, y, x + cw, y + ch),
        ))

    return fields


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _deduplicate(fields: list[DetectedField], w: int, h: int) -> list[DetectedField]:
    """
    Remove duplicate detections from multiple passes.
    Priority order: checkbox > underline > box (more specific type wins).
    Within same type: high confidence wins over low.
    Also flags detections that are very close to another as low confidence.
    """
    if not fields:
        return fields

    # Sort: checkboxes first (most specific), then by confidence
    _type_priority = {"checkbox": 0, "underline": 1, "box": 2}
    fields.sort(key=lambda f: (
        _type_priority.get(f.field_type, 9),
        0 if f.confidence == "high" else 1
    ))

    kept: list[DetectedField] = []
    for f in fields:
        x1, y1, x2, y2 = f.bbox_px
        dominated = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k.bbox_px
            if _iou((x1, y1, x2, y2), (kx1, ky1, kx2, ky2)) > 0.4:
                dominated = True
                break
        if not dominated:
            kept.append(f)

    # Flag detections that are very close to another (possible partial duplicate)
    for i, f in enumerate(kept):
        for j, g in enumerate(kept):
            if i == j:
                continue
            dist = _min_edge_distance(f.bbox_px, g.bbox_px)
            if dist < OVERLAP_PROXIMITY_PX:
                f.confidence = "low"
                break

    return kept


def _normalize(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple:
    return (
        round(x1 / w, 4),
        round(y1 / h, 4),
        round(x2 / w, 4),
        round(y2 / h, 4),
    )


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over Union for two (x1,y1,x2,y2) pixel boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _min_edge_distance(a: tuple, b: tuple) -> float:
    """Minimum distance between the edges of two rectangles."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(ax1, bx1) - min(ax2, bx2))
    dy = max(0, max(ay1, by1) - min(ay2, by2))
    return float(dx + dy)


def _overlaps_any(
    x1: int, y1: int, x2: int, y2: int,
    existing: list[tuple],
    threshold: float = 0.5,
) -> bool:
    for (ex1, ey1, ex2, ey2) in existing:
        if _iou((x1, y1, x2, y2), (ex1, ey1, ex2, ey2)) >= threshold:
            return True
    return False
