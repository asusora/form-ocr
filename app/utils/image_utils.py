"""图像处理工具。"""

from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np

from app.models.common import BBox


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """将图像转换为灰度图。"""

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise_and_enhance(image: np.ndarray) -> np.ndarray:
    """执行去噪与 CLAHE 增强。"""

    gray = to_grayscale(image)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)


def detect_skew_angle(image: np.ndarray) -> float:
    """检测页面倾斜角度。"""

    gray = to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=max(200, gray.shape[1] // 4),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -15.0 <= angle <= 15.0:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """按指定角度旋转图像。"""

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_page(image: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """执行页面预处理并返回处理图、旋转角和是否应用矫正。"""

    enhanced = denoise_and_enhance(image)
    angle = detect_skew_angle(enhanced)
    if abs(angle) < 0.2:
        return enhanced, 0.0, False

    corrected = rotate_image(enhanced, -angle)
    return corrected, round(-angle, 4), True


def polygon_to_bbox(polygon: list[list[float]] | np.ndarray, width: int, height: int) -> BBox:
    """将 OCR 多边形转换为归一化矩形框。"""

    points = np.asarray(polygon, dtype=np.float32)
    x_min = float(np.min(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    x_max = float(np.max(points[:, 0]))
    y_max = float(np.max(points[:, 1]))
    return BBox(
        x=max(0.0, min(1.0, x_min / width)),
        y=max(0.0, min(1.0, y_min / height)),
        w=max(0.0, min(1.0, (x_max - x_min) / width)),
        h=max(0.0, min(1.0, (y_max - y_min) / height)),
    )


def expand_bbox(bbox: BBox, x_padding: float, y_padding: float) -> BBox:
    """对归一化框做外扩。"""

    return expand_bbox_asymmetric(
        bbox,
        left_padding=x_padding,
        top_padding=y_padding,
        right_padding=x_padding,
        bottom_padding=y_padding,
    )


def expand_bbox_asymmetric(
    bbox: BBox,
    *,
    left_padding: float,
    top_padding: float,
    right_padding: float,
    bottom_padding: float,
) -> BBox:
    """按四个方向分别扩张归一化框。"""

    x = max(0.0, bbox.x - left_padding)
    y = max(0.0, bbox.y - top_padding)
    right = min(1.0, bbox.x + bbox.w + right_padding)
    bottom = min(1.0, bbox.y + bbox.h + bottom_padding)
    return BBox(x=x, y=y, w=max(0.0, right - x), h=max(0.0, bottom - y))


def rotate_bbox(bbox: BBox, width: int, height: int, angle_degrees: float) -> BBox:
    """将归一化矩形框按页面中心旋转，并返回旋转后外接矩形。"""

    if width <= 0 or height <= 0 or abs(angle_degrees) < 1e-6:
        return BBox(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h)

    center_x = width / 2.0
    center_y = height / 2.0
    radians = math.radians(angle_degrees)
    cos_value = math.cos(radians)
    sin_value = math.sin(radians)

    corners = (
        (bbox.x * width, bbox.y * height),
        ((bbox.x + bbox.w) * width, bbox.y * height),
        ((bbox.x + bbox.w) * width, (bbox.y + bbox.h) * height),
        (bbox.x * width, (bbox.y + bbox.h) * height),
    )

    rotated_points: list[tuple[float, float]] = []
    for point_x, point_y in corners:
        translated_x = point_x - center_x
        translated_y = point_y - center_y
        rotated_x = translated_x * cos_value + translated_y * sin_value + center_x
        rotated_y = -translated_x * sin_value + translated_y * cos_value + center_y
        rotated_points.append((rotated_x, rotated_y))

    min_x = max(0.0, min(point[0] for point in rotated_points))
    min_y = max(0.0, min(point[1] for point in rotated_points))
    max_x = min(float(width), max(point[0] for point in rotated_points))
    max_y = min(float(height), max(point[1] for point in rotated_points))
    max_x = max(min_x, max_x)
    max_y = max(min_y, max_y)

    return BBox(
        x=min_x / width,
        y=min_y / height,
        w=(max_x - min_x) / width,
        h=(max_y - min_y) / height,
    )


def map_bbox_to_original_space(bbox: BBox, width: int, height: int, rotation_degree: float) -> BBox:
    """将预处理图坐标系中的框逆变换回原图坐标系。"""

    return rotate_bbox(bbox, width, height, -rotation_degree)


def bbox_to_pixels(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    """将归一化框转换为像素坐标。"""

    x1 = max(0, min(width - 1, int(round(bbox.x * width))))
    y1 = max(0, min(height - 1, int(round(bbox.y * height))))
    x2 = max(x1 + 1, min(width, int(round((bbox.x + bbox.w) * width))))
    y2 = max(y1 + 1, min(height, int(round((bbox.y + bbox.h) * height))))
    return x1, y1, x2, y2


def crop_by_bbox(image: np.ndarray, bbox: BBox) -> np.ndarray:
    """根据归一化框裁切图像。"""

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox_to_pixels(bbox, width, height)
    return image[y1:y2, x1:x2]


def compute_ink_ratio(image: np.ndarray) -> float:
    """计算墨迹填充率。"""

    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    total = binary.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(binary) / total)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    """返回归一化框中心点。"""

    return bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0


def bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """计算两个归一化框的 IoU。"""

    left = max(bbox_a.x, bbox_b.x)
    top = max(bbox_a.y, bbox_b.y)
    right = min(bbox_a.x + bbox_a.w, bbox_b.x + bbox_b.w)
    bottom = min(bbox_a.y + bbox_a.h, bbox_b.y + bbox_b.h)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = bbox_a.w * bbox_a.h + bbox_b.w * bbox_b.h - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def deduplicate_bboxes(boxes: Iterable[BBox], threshold: float = 0.65) -> list[BBox]:
    """按 IoU 对框去重。"""

    deduplicated: list[BBox] = []
    for bbox in boxes:
        if any(bbox_iou(bbox, existing) >= threshold for existing in deduplicated):
            continue
        deduplicated.append(bbox)
    return deduplicated
