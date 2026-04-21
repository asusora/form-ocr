"""阶段 1 左通道实现。"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from app.core.config import Settings
from app.models.common import AnchorText, BBox, FieldType, RouteType
from app.models.stages import FieldCandidate, LeftChannelOutput, LeftChannelPageOutput, RouteDecision, Stage0Output
from app.repositories.task_repository import TaskRepository
from app.services.ocr_service import BaseOcrEngine
from app.utils.id_utils import generate_field_id
from app.utils.image_utils import (
    bbox_center,
    bbox_iou,
    bbox_to_pixels,
    compute_ink_ratio,
    crop_by_bbox,
    deduplicate_bboxes,
    expand_bbox,
    expand_bbox_asymmetric,
    to_grayscale,
)


class LeftChannelService:
    """负责字段候选检测、裁切和基础 OCR。"""

    DATE_KEYWORDS = ("date", "日期", "年月日", "出生", "出生日期")
    SIGNATURE_KEYWORDS = ("signature", "签名", "簽名", "signed by", "applicant signature")

    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        ocr_engine: BaseOcrEngine,
    ) -> None:
        """注入依赖。"""

        self.settings = settings
        self.repository = repository
        self.ocr_engine = ocr_engine

    def run(self, stage0_output: Stage0Output, route_decision: RouteDecision) -> LeftChannelOutput:
        """执行左通道检测。"""

        anchors_by_page: dict[int, list[AnchorText]] = defaultdict(list)
        page_lookup = {page.page_id: page.page_index for page in stage0_output.pages}
        for anchor in stage0_output.anchors:
            page_index = page_lookup.get(anchor.page_id)
            if page_index is not None:
                anchors_by_page[page_index].append(anchor)

        route_map = {page_route.page_index: page_route for page_route in route_decision.page_routes}
        page_results: list[LeftChannelPageOutput] = []
        for page in stage0_output.pages:
            origin_image = cv2.imread(str(self.repository.url_to_path(page.origin_image_url)))
            preprocessed_image = cv2.imread(str(self.repository.url_to_path(page.preprocessed_image_url)), cv2.IMREAD_GRAYSCALE)
            if origin_image is None or preprocessed_image is None:
                page_results.append(LeftChannelPageOutput(page_index=page.page_index, field_candidates=[]))
                continue

            page_route = route_map.get(page.page_index)
            field_candidates = self._process_page(
                task_id=stage0_output.task_id,
                page_index=page.page_index,
                origin_image=origin_image,
                preprocessed_image=preprocessed_image,
                anchors=anchors_by_page.get(page.page_index, []),
                page_route=page_route,
            )
            page_results.append(
                LeftChannelPageOutput(
                    page_index=page.page_index,
                    field_candidates=field_candidates,
                )
            )

        return LeftChannelOutput(task_id=stage0_output.task_id, pages=page_results)

    def _process_page(
        self,
        *,
        task_id: str,
        page_index: int,
        origin_image: np.ndarray,
        preprocessed_image: np.ndarray,
        anchors: list[AnchorText],
        page_route,
    ) -> list[FieldCandidate]:
        """处理单页候选框。"""

        raw_candidates: list[dict] = []
        if page_route and page_route.route_type in {RouteType.TEMPLATE_HIT, RouteType.TEMPLATE_PARTIAL}:
            for region in page_route.known_field_regions:
                raw_candidates.append(
                    {
                        "bbox": region.bbox_relative,
                        "field_type": region.field_type,
                        "detector_source": "template_seed",
                        "canonical_key_id": region.canonical_key_id,
                        "key_hint": region.key,
                        "key_hint_en": region.key_en,
                    }
                )

        if page_route is None or page_route.route_type in {RouteType.TEMPLATE_UNKNOWN, RouteType.TEMPLATE_PARTIAL}:
            for bbox in self._detect_line_bboxes(preprocessed_image):
                raw_candidates.append(
                    {
                        "bbox": bbox,
                        "field_type": None,
                        "detector_source": "morph_line_detector",
                        "canonical_key_id": None,
                        "key_hint": None,
                        "key_hint_en": None,
                    }
                )
            for bbox in self._detect_checkbox_bboxes(preprocessed_image):
                raw_candidates.append(
                    {
                        "bbox": bbox,
                        "field_type": FieldType.CHECKBOX,
                        "detector_source": "contour_checkbox_detector",
                        "canonical_key_id": None,
                        "key_hint": None,
                        "key_hint_en": None,
                    }
                )
            for bbox in self._detect_box_bboxes(preprocessed_image):
                raw_candidates.append(
                    {
                        "bbox": bbox,
                        "field_type": None,
                        "detector_source": "contour_box_detector",
                        "canonical_key_id": None,
                        "key_hint": None,
                        "key_hint_en": None,
                    }
                )

        deduped: list[dict] = []
        for candidate in sorted(raw_candidates, key=lambda item: (item["bbox"].y, item["bbox"].x)):
            if any(bbox_iou(candidate["bbox"], existing["bbox"]) >= 0.75 for existing in deduped):
                continue
            deduped.append(candidate)

        field_candidates: list[FieldCandidate] = []
        for field_index, candidate in enumerate(deduped):
            anchor_text = candidate["key_hint"] or self._find_nearest_anchor_text(candidate["bbox"], anchors)
            field_type = candidate["field_type"] or self._infer_field_type(anchor_text)
            field_id = generate_field_id(task_id, page_index, field_index)
            field = self._finalize_candidate(
                field_id=field_id,
                task_id=task_id,
                page_index=page_index,
                field_type=field_type,
                bbox=candidate["bbox"],
                detector_source=candidate["detector_source"],
                origin_image=origin_image,
                preprocessed_image=preprocessed_image,
                line_anchor_text=anchor_text,
                canonical_key_id=candidate["canonical_key_id"],
                key_hint=candidate["key_hint"],
                key_hint_en=candidate["key_hint_en"],
            )
            field_candidates.append(field)

        return sorted(field_candidates, key=lambda item: (item.bbox.y, item.bbox.x))

    # 真实下划线高度 1-8 px；超出即为文字块底边或多行合并残影。
    UNDERLINE_MIN_H_PX = 1
    UNDERLINE_MAX_H_PX = 8

    def _detect_line_bboxes(self, image: np.ndarray) -> list[BBox]:
        """检测横线填写区域。"""

        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        width = gray.shape[1]
        height = gray.shape[0]
        char_height_px = self._estimate_char_height(binary, width, height)
        kernel_length = max(25, width // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # 仅做 3x3 轻度膨胀合并断裂；原先 (7,3) 会把下划线与上方文字连成一坨。
        dilated = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[BBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            width_ratio = w / width
            height_ratio = h / height
            if width_ratio < self.settings.line_min_width_ratio:
                continue
            if width_ratio > self.settings.line_max_width_ratio:
                continue
            if height_ratio > self.settings.line_max_height_ratio:
                continue
            # 绝对高度上限：真实下划线不会高于 ~8 px；高于此即为文字块/表格底边残影。
            if not (self.UNDERLINE_MIN_H_PX <= h <= self.UNDERLINE_MAX_H_PX):
                continue
            if y < height * 0.02 or y > height * 0.98:
                continue
            line_bbox = BBox(x=x / width, y=y / height, w=w / width, h=h / height)
            boxes.append(
                self._expand_line_bbox_for_display(
                    binary=binary,
                    bbox=line_bbox,
                    width=width,
                    height=height,
                    char_height_px=char_height_px,
                )
            )
        return deduplicate_bboxes(boxes)

    def _estimate_char_height(self, binary: np.ndarray, width: int, height: int) -> int:
        """估算页面中的典型字符高度，避免下划线字段的上移幅度失真。"""

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_char_height = max(4, int(height * 0.005))
        max_char_height = max(min_char_height, int(height * 0.03))
        min_char_width = max(2, int(width * 0.001))
        max_char_width = max(min_char_width, int(width * 0.08))

        candidate_heights: list[int] = []
        for label_index in range(1, num_labels):
            component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if not (min_char_height <= component_height <= max_char_height):
                continue
            if not (min_char_width <= component_width <= max_char_width):
                continue
            aspect_ratio = component_width / max(component_height, 1)
            if not (0.2 <= aspect_ratio <= 4.0):
                continue
            candidate_heights.append(component_height)

        if len(candidate_heights) < 10:
            return max(8, int(height * 0.018))
        return int(np.median(candidate_heights))

    def _expand_line_bbox_for_display(
        self,
        *,
        binary: np.ndarray,
        bbox: BBox,
        width: int,
        height: int,
        char_height_px: int,
    ) -> BBox:
        """将下划线框上移到填写区，避免预览框压在线条正中间。"""

        x1, y_top_line, x2, y_bottom_line = bbox_to_pixels(bbox, width, height)
        search_window = max(12, int(char_height_px * 1.8))
        ceil_y = max(0, y_top_line - search_window)
        expanded_top = ceil_y

        roi = binary[ceil_y:y_top_line, x1:x2]
        if roi.size > 0 and roi.shape[0] > 0 and roi.shape[1] > 0:
            row_counts = np.sum(roi > 0, axis=1)
            ink_threshold = max(3, int(0.02 * max(1, x2 - x1)))
            text_row_indices = np.where(row_counts > ink_threshold)[0]
            if len(text_row_indices) > 0:
                gap_threshold = max(2, char_height_px // 3)
                gaps = np.where(np.diff(text_row_indices) > gap_threshold)[0]
                cluster = text_row_indices[gaps[-1] + 1:] if len(gaps) > 0 else text_row_indices
                cluster_top = ceil_y + int(cluster[0])
                cluster_bottom = ceil_y + int(cluster[-1])
                if y_top_line - cluster_bottom <= max(4, int(char_height_px * 1.1)):
                    cluster_height = max(1, cluster_bottom - cluster_top)
                    padding = max(2, int(cluster_height * 0.15))
                    expanded_top = max(ceil_y, cluster_top - padding)

        expanded_bottom = min(height, y_bottom_line + max(1, int(char_height_px * 0.15)))
        return BBox(
            x=x1 / width,
            y=expanded_top / height,
            w=max(0.0, (x2 - x1) / width),
            h=max(0.0, (expanded_bottom - expanded_top) / height),
        )

    # 300 DPI 下实测表单 checkbox ≈ 49-55 px；字符高度 < 30 px。
    # 用 40 px 作为最小边长，足以剔除全部正文汉字与标点。
    CHECKBOX_MIN_SIDE_PX = 40
    # 空心填充率：checkbox 只有边框着墨，填充率应该很低。
    # 实心字符（回、田、国…）的 bounding box 填充率 > 0.45。
    CHECKBOX_MIN_FILL = 0.05
    CHECKBOX_MAX_FILL = 0.40

    def _detect_checkbox_bboxes(self, image: np.ndarray) -> list[BBox]:
        """检测勾选框。"""

        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # RETR_LIST 不含嵌套层级，配合下方位置去重抑制内/外轮廓重复。
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        width = gray.shape[1]
        height = gray.shape[0]

        boxes: list[BBox] = []
        seen_positions: set[tuple[int, int]] = set()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.settings.checkbox_min_area or area > self.settings.checkbox_max_area:
                continue
            ratio = w / max(h, 1)
            if not 0.75 <= ratio <= 1.25:
                continue
            if w < self.CHECKBOX_MIN_SIDE_PX or h < self.CHECKBOX_MIN_SIDE_PX:
                continue
            if x < 2 or y < 2 or x + w >= width - 2 or y + h >= height - 2:
                continue

            roi = binary[y:y + h, x:x + w]
            fill_ratio = float(np.count_nonzero(roi)) / (area + 1e-6)
            if not (self.CHECKBOX_MIN_FILL <= fill_ratio <= self.CHECKBOX_MAX_FILL):
                continue

            # 多边形近似：真实 checkbox 边框 ≈ 矩形（4-8 角）；圆/弧/复杂字符形被排除。
            peri = cv2.arcLength(contour, True)
            if peri < 1:
                continue
            approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue

            # 内部留白：bbox 内缩 ~1/6 后的核心区域必须接近空白。
            # 汉字（回、田、国）/ 数字（0、8、9）虽空心但笔画延伸到内部，会被此关卡卡掉。
            margin = max(2, min(w, h) // 6)
            interior = roi[margin:h - margin, margin:w - margin]
            if interior.size > 0:
                interior_fill = float(np.count_nonzero(interior)) / (interior.size + 1e-6)
                if interior_fill > 0.15:
                    continue

            # 量化到 6 px 网格抑制近位重复轮廓（同一方框的内/外描边）。
            pos_key = (x // 6, y // 6)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            boxes.append(BBox(x=x / width, y=y / height, w=w / width, h=h / height))
        return deduplicate_bboxes(boxes)

    # 矩形输入框（Part 3 的 REW / REC 字段）——宽 > 高，面积远大于 checkbox。
    BOX_MIN_AREA_FRAC = 0.0010   # ≥ 0.10% 页面面积
    BOX_MAX_AREA_FRAC = 0.25     # ≤ 25% 排除全页内容块
    BOX_MIN_WIDTH_FRAC = 0.03    # ≥ 3% 页宽
    BOX_MAX_WIDTH_FRAC = 0.95    # 排除整页边框
    BOX_MIN_HEIGHT_FRAC = 0.008  # ≥ 0.8% 页高，排除薄片噪声
    BOX_MAX_OVERALL_FILL = 0.35  # 整体墨迹占比 ≤ 35%（真实空框很稀疏）
    BOX_MAX_INTERIOR_FILL = 0.12 # 内缩 1/7 后的核心区填充 ≤ 12%（排除被边框包围的文字块）

    def _detect_box_bboxes(self, image: np.ndarray) -> list[BBox]:
        """检测矩形输入框（比 checkbox 大、比整页小的带边框字段）。"""

        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 闭合：合并因扫描/打印断裂的边框线段
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        # RETR_EXTERNAL：只取最外层轮廓，避免重复捕获嵌套结构
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        width = gray.shape[1]
        height = gray.shape[0]
        page_area = width * height
        min_h_px = max(12, int(height * self.BOX_MIN_HEIGHT_FRAC))
        checkbox_max_side = int(width * 0.06)  # 与 checkbox 检测的尺寸上限对齐

        boxes: list[BBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # 跳过 checkbox 尺寸（小方框由 _detect_checkbox_bboxes 负责）
            if w <= checkbox_max_side and h <= checkbox_max_side:
                aspect = w / max(h, 1)
                if 0.75 <= aspect <= 1.33:
                    continue

            if h < min_h_px:
                continue
            if not (self.BOX_MIN_AREA_FRAC * page_area <= area <= self.BOX_MAX_AREA_FRAC * page_area):
                continue
            if not (self.BOX_MIN_WIDTH_FRAC * width <= w <= self.BOX_MAX_WIDTH_FRAC * width):
                continue
            # 排除竖向条（竖向装饰线、左侧索引条等）
            if h > w * 1.2:
                continue
            # 边缘排除
            if x < 2 or y < 2 or x + w >= width - 2 or y + h >= height - 2:
                continue

            # 多边形近似：干净矩形应为 4-6 角
            peri = cv2.arcLength(contour, True)
            if peri < 1:
                continue
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) < 4 or len(approx) > 6:
                continue

            roi = binary[y:y + h, x:x + w]
            # 整体填充率：空框应稀疏
            overall_fill = float(np.count_nonzero(roi)) / (area + 1e-6)
            if overall_fill > self.BOX_MAX_OVERALL_FILL:
                continue

            # 内部核心区填充率：剔除被边框包围的文字段落（如表格标题行）
            mx = max(2, w // 7)
            my = max(2, h // 7)
            interior = roi[my:h - my, mx:w - mx]
            if interior.size > 0:
                interior_fill = float(np.count_nonzero(interior)) / (interior.size + 1e-6)
                if interior_fill > self.BOX_MAX_INTERIOR_FILL:
                    continue

            boxes.append(BBox(x=x / width, y=y / height, w=w / width, h=h / height))
        return deduplicate_bboxes(boxes)

    def _find_nearest_anchor_text(self, bbox: BBox, anchors: list[AnchorText]) -> str | None:
        """查找与字段最接近的锚点文本。"""

        if not anchors:
            return None

        field_cx, field_cy = bbox_center(bbox)
        best_anchor: AnchorText | None = None
        best_score = float("inf")
        for anchor in anchors:
            anchor_cx, anchor_cy = bbox_center(anchor.bbox)
            horizontal_penalty = 0.0 if anchor_cx <= field_cx else 0.25
            vertical_penalty = 0.0 if anchor_cy <= field_cy + 0.02 else 0.15
            distance = ((field_cx - anchor_cx) ** 2 + (field_cy - anchor_cy) ** 2) ** 0.5
            score = distance + horizontal_penalty + vertical_penalty
            if score < best_score:
                best_score = score
                best_anchor = anchor
        if best_anchor is None or best_score > 0.45:
            return None
        return best_anchor.text

    def _infer_field_type(self, anchor_text: str | None) -> FieldType:
        """根据附近锚点文本推断字段类型。"""

        normalized = (anchor_text or "").strip().lower()
        if any(keyword in normalized for keyword in self.SIGNATURE_KEYWORDS):
            return FieldType.SIGNATURE
        if any(keyword in normalized for keyword in self.DATE_KEYWORDS):
            return FieldType.DATE
        return FieldType.TEXT

    def _finalize_candidate(
        self,
        *,
        field_id: str,
        task_id: str,
        page_index: int,
        field_type: FieldType,
        bbox: BBox,
        detector_source: str,
        origin_image: np.ndarray,
        preprocessed_image: np.ndarray,
        line_anchor_text: str | None,
        canonical_key_id: str | None,
        key_hint: str | None,
        key_hint_en: str | None,
    ) -> FieldCandidate:
        """裁切字段并生成最终候选对象。"""

        if field_type == FieldType.CHECKBOX:
            display_bbox = bbox
            crop_bbox = expand_bbox(bbox, 0.003, 0.003)
        elif detector_source == "morph_line_detector":
            display_bbox = bbox
            crop_bbox = expand_bbox_asymmetric(
                bbox,
                left_padding=0.008,
                top_padding=0.010,
                right_padding=0.008,
                bottom_padding=0.004,
            )
        else:
            display_bbox = bbox
            crop_bbox = expand_bbox(bbox, 0.008, 0.015)

        crop_origin = crop_by_bbox(origin_image, crop_bbox)
        crop_preprocessed = crop_by_bbox(preprocessed_image, crop_bbox)
        crop_path = self.repository.build_artifact_path(task_id, "fields", f"{field_id}.png")
        cv2.imwrite(str(crop_path), crop_origin)

        ink_ratio = round(compute_ink_ratio(crop_preprocessed), 4)
        ocr_text = None
        ocr_confidence = None
        is_checked = None
        is_filled = ink_ratio >= self.settings.text_filled_ink_threshold

        if field_type in {FieldType.TEXT, FieldType.DATE, FieldType.HANDWRITING}:
            text, confidence = self.ocr_engine.read_text(crop_preprocessed)
            ocr_text = text or None
            ocr_confidence = round(confidence, 4) if text else 0.0
            if field_type == FieldType.TEXT and is_filled and (ocr_confidence or 0.0) < 0.45:
                field_type = FieldType.HANDWRITING
            is_filled = bool((ocr_text or "").strip()) or is_filled
        elif field_type == FieldType.CHECKBOX:
            checkbox_crop = crop_preprocessed
            if checkbox_crop.size > 0:
                checkbox_crop = checkbox_crop[2:-2, 2:-2] if checkbox_crop.shape[0] > 4 and checkbox_crop.shape[1] > 4 else checkbox_crop
            fill_ratio = compute_ink_ratio(checkbox_crop)
            is_checked = fill_ratio >= self.settings.checkbox_fill_threshold
            ink_ratio = round(fill_ratio, 4)
            is_filled = is_checked
        else:
            is_filled = ink_ratio >= self.settings.text_filled_ink_threshold

        return FieldCandidate(
            field_id=field_id,
            task_id=task_id,
            page_index=page_index,
            field_type=field_type,
            bbox=display_bbox,
            crop_bbox=crop_bbox,
            detector_source=detector_source,
            crop_image_url=self.repository.path_to_url(crop_path),
            ocr_text=ocr_text,
            ocr_confidence=ocr_confidence,
            is_filled=is_filled,
            ink_ratio=ink_ratio,
            line_anchor_text=line_anchor_text,
            is_checked=is_checked,
            canonical_key_id=canonical_key_id,
            key_hint=key_hint,
            key_hint_en=key_hint_en,
        )
