"""阶段 1 左通道实现。"""

from __future__ import annotations

from collections import defaultdict
import re

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
    expand_bbox_asymmetric,
    map_bbox_to_original_space,
    to_grayscale,
)


class LeftChannelService:
    """负责字段候选检测、裁切和基础 OCR。"""

    DATE_KEYWORDS = ("date", "日期", "年月日", "出生", "出生日期")
    SIGNATURE_KEYWORDS = ("signature", "签名", "簽名", "signed by", "applicant signature")
    FIELD_CROP_LEFT_RIGHT_PADDING = 0.008
    FIELD_CROP_TOP_BOTTOM_PADDING = 0.002

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
                rotation_degree=page.rotation_degree,
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
        rotation_degree: float = 0.0,
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
            for bbox in self._detect_option_group_bboxes(preprocessed_image, anchors):
                raw_candidates.append(
                    {
                        "bbox": bbox,
                        "field_type": FieldType.TEXT,
                        "detector_source": "anchor_option_detector",
                        "canonical_key_id": None,
                        "key_hint": None,
                        "key_hint_en": None,
                    }
                )
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
            nearest_anchor = None if candidate["key_hint"] else self._find_nearest_anchor(candidate["bbox"], anchors)
            if candidate["detector_source"] == "morph_line_detector" and self._should_reject_non_field_line_candidate(
                bbox=candidate["bbox"],
                anchors=anchors,
                nearest_anchor=nearest_anchor,
            ):
                continue
            if candidate["detector_source"] == "morph_line_detector" and self._is_text_embedded_short_line_candidate(
                bbox=candidate["bbox"],
                anchors=anchors,
            ):
                continue
            anchor_text = candidate["key_hint"] or (nearest_anchor.text if nearest_anchor is not None else None)
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
                rotation_degree=rotation_degree,
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
    SHORT_UNDERLINE_MIN_WIDTH_RATIO = 0.012
    SHORT_UNDERLINE_MAX_WIDTH_RATIO = 0.16
    SHORT_UNDERLINE_MIN_WIDTH_PX = 18
    SHORT_UNDERLINE_KERNEL_MIN_PX = 12
    SHORT_UNDERLINE_KERNEL_MAX_PX = 24
    SHORT_UNDERLINE_KERNEL_CHAR_HEIGHT_RATIO = 1.6
    UNDERLINE_MIN_ASPECT_RATIO = 6.0
    DATE_SEGMENT_MAX_GAP_RATIO = 2.2
    DATE_SEGMENT_MAX_WIDTH_RATIO = 8.5
    UNDERLINE_BELOW_DENSITY_THRESHOLD = 0.12
    SEGMENT_GROUP_MAX_TOTAL_WIDTH_RATIO = 0.36
    SEGMENT_SEPARATOR_MAX_COMPONENTS = 4
    SEGMENT_SEPARATOR_MAX_DENSITY = 0.55
    HEADER_DECORATION_MIN_WIDTH_RATIO = 0.45
    HEADER_DECORATION_MAX_Y_RATIO = 0.12
    HEADER_ANCHOR_OVERLAP_RATIO = 0.30
    HEADER_ANCHOR_MAX_GAP_RATIO = 1.6
    TOP_SHORT_STROKE_MAX_Y_RATIO = 0.14
    TOP_SHORT_STROKE_MIN_COMPONENT_WIDTH_RATIO = 0.50
    TOP_SHORT_STROKE_MIN_COMPONENT_HEIGHT_RATIO = 0.45
    TEXT_EMBEDDED_SHORT_LINE_MAX_WIDTH_RATIO = 0.20
    TEXT_EMBEDDED_SHORT_LINE_MAX_CENTER_Y_RATIO = 0.84
    TEXT_EMBEDDED_SHORT_LINE_ANCHOR_X_PADDING_RATIO = 0.004
    TEXT_EMBEDDED_SHORT_LINE_ANCHOR_TOP_PADDING_RATIO = 0.002
    TEXT_EMBEDDED_SHORT_LINE_ANCHOR_BOTTOM_PADDING_RATIO = 0.12
    OPTION_GROUP_PATTERN = re.compile(r"[\[\u3010\u3014]\s*([^\]\u3011\u3015]{1,80})\s*[\]\u3011\u3015]")
    OPTION_GROUP_MIN_TEXT_UNITS = 4.0
    OPTION_GROUP_MAX_TEXT_UNITS = 42.0
    OPTION_GROUP_MAX_SLASH_COUNT = 3
    OPTION_GROUP_MIN_WIDTH_RATIO = 0.02
    OPTION_GROUP_MAX_WIDTH_RATIO = 0.30

    def _detect_line_bboxes(self, image: np.ndarray) -> list[BBox]:
        """检测横线填写区域。"""

        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        width = gray.shape[1]
        height = gray.shape[0]
        char_height_px = self._estimate_char_height(binary, width, height)
        short_kernel_length = max(
            self.SHORT_UNDERLINE_KERNEL_MIN_PX,
            min(
                self.SHORT_UNDERLINE_KERNEL_MAX_PX,
                int(char_height_px * self.SHORT_UNDERLINE_KERNEL_CHAR_HEIGHT_RATIO),
            ),
        )
        short_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (short_kernel_length, 1))
        short_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, short_kernel, iterations=1)
        short_line_boxes = self._extract_line_bboxes_from_mask(
            mask=short_opened,
            binary=binary,
            width=width,
            height=height,
            char_height_px=char_height_px,
            min_width_ratio=min(self.settings.line_min_width_ratio, self.SHORT_UNDERLINE_MIN_WIDTH_RATIO),
            max_width_ratio=min(self.settings.line_max_width_ratio, self.SHORT_UNDERLINE_MAX_WIDTH_RATIO),
            require_blank_below=True,
        )
        kernel_length = max(25, width // 30)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        long_line_boxes = self._extract_line_bboxes_from_mask(
            mask=dilated,
            binary=binary,
            width=width,
            height=height,
            char_height_px=char_height_px,
            min_width_ratio=self.settings.line_min_width_ratio,
            max_width_ratio=self.settings.line_max_width_ratio,
            require_blank_below=False,
        )
        raw_boxes = deduplicate_bboxes(long_line_boxes + short_line_boxes, threshold=0.55)
        merged_boxes = self._merge_date_like_line_groups(raw_boxes, binary, width, height, char_height_px)

        display_boxes: list[BBox] = []
        for line_bbox in merged_boxes:
            display_boxes.append(
                self._expand_line_bbox_for_display(
                    binary=binary,
                    bbox=line_bbox,
                    width=width,
                    height=height,
                    char_height_px=char_height_px,
                )
            )
        return deduplicate_bboxes(display_boxes)

    def _extract_line_bboxes_from_mask(
        self,
        *,
        mask: np.ndarray,
        binary: np.ndarray,
        width: int,
        height: int,
        char_height_px: int,
        min_width_ratio: float,
        max_width_ratio: float,
        require_blank_below: bool,
    ) -> list[BBox]:
        """从给定线条掩码中提取下划线候选框。"""

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[BBox] = []
        min_width_px = max(self.SHORT_UNDERLINE_MIN_WIDTH_PX, int(char_height_px * 1.2))

        for contour in contours:
            x, y, contour_width, contour_height = cv2.boundingRect(contour)
            width_ratio = contour_width / width
            height_ratio = contour_height / height
            if width_ratio < min_width_ratio or width_ratio > max_width_ratio:
                continue
            if height_ratio > self.settings.line_max_height_ratio:
                continue
            if contour_width < min_width_px:
                continue
            if contour_width / max(contour_height, 1) < self.UNDERLINE_MIN_ASPECT_RATIO:
                continue
            if not (self.UNDERLINE_MIN_H_PX <= contour_height <= self.UNDERLINE_MAX_H_PX):
                continue
            if y < height * 0.02 or y > height * 0.98:
                continue
            if require_blank_below and not self._has_blank_band_below(
                binary=binary,
                x=x,
                y=y,
                width_px=contour_width,
                line_height_px=contour_height,
                image_height=height,
                char_height_px=char_height_px,
            ):
                continue
            if require_blank_below and self._is_embedded_top_short_stroke(
                binary=binary,
                x=x,
                y=y,
                width_px=contour_width,
                line_height_px=contour_height,
                image_height=height,
                char_height_px=char_height_px,
            ):
                continue
            boxes.append(
                BBox(
                    x=x / width,
                    y=y / height,
                    w=contour_width / width,
                    h=contour_height / height,
                )
            )
        return boxes

    def _has_blank_band_below(
        self,
        *,
        binary: np.ndarray,
        x: int,
        y: int,
        width_px: int,
        line_height_px: int,
        image_height: int,
        char_height_px: int,
    ) -> bool:
        """判断线条下方是否基本为空白，用于区分短下划线与正文横笔。"""

        band_top = min(image_height, y + line_height_px)
        band_bottom = min(image_height, band_top + max(2, char_height_px // 3))
        if band_bottom <= band_top:
            return True
        band = binary[band_top:band_bottom, x:x + width_px]
        if band.size == 0:
            return True
        density = float(np.count_nonzero(band)) / float(band.size)
        return density <= self.UNDERLINE_BELOW_DENSITY_THRESHOLD

    def _is_embedded_top_short_stroke(
        self,
        *,
        binary: np.ndarray,
        x: int,
        y: int,
        width_px: int,
        line_height_px: int,
        image_height: int,
        char_height_px: int,
    ) -> bool:
        """判断顶部短横线是否更像 logo 或装饰图形中的笔画。"""

        if image_height <= 0:
            return False

        line_center_y_ratio = (y + line_height_px / 2.0) / float(image_height)
        if line_center_y_ratio > self.TOP_SHORT_STROKE_MAX_Y_RATIO:
            return False

        band_height = max(4, int(char_height_px * 0.9))
        band_top = max(0, y - band_height)
        band_bottom = y
        if band_bottom <= band_top:
            return False

        trim_padding = max(0, int(width_px * 0.1))
        region_left = x + trim_padding
        region_right = x + width_px - trim_padding
        if region_right - region_left < max(6, width_px // 2):
            region_left = x
            region_right = x + width_px

        upper_region = binary[band_top:band_bottom, region_left:region_right]
        if upper_region.size == 0:
            return False

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(upper_region, connectivity=8)
        min_component_width = max(6, int(upper_region.shape[1] * self.TOP_SHORT_STROKE_MIN_COMPONENT_WIDTH_RATIO))
        min_component_height = max(4, int(upper_region.shape[0] * self.TOP_SHORT_STROKE_MIN_COMPONENT_HEIGHT_RATIO))

        for label_index in range(1, num_labels):
            component_area = int(stats[label_index, cv2.CC_STAT_AREA])
            if component_area < 8:
                continue
            component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if component_width >= min_component_width and component_height >= min_component_height:
                return True
        return False

    def _detect_option_group_bboxes(self, image: np.ndarray, anchors: list[AnchorText]) -> list[BBox]:
        """基于 OCR 锚点检测括号包裹的选项组，例如 [Y/N]、【是/否】。"""

        if not anchors:
            return []

        gray = to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        height, width = gray.shape[:2]
        boxes: list[BBox] = []

        for anchor in anchors:
            text = (anchor.text or "").strip()
            if not text:
                continue
            for match in self.OPTION_GROUP_PATTERN.finditer(text):
                content = match.group(1)
                if not self._is_option_group_content(content):
                    continue
                estimated_bbox = self._estimate_anchor_text_span_bbox(
                    anchor_bbox=anchor.bbox,
                    full_text=text,
                    start_index=match.start(),
                    end_index=match.end(),
                    image_width=width,
                    image_height=height,
                )
                refined_bbox = self._refine_text_span_bbox(
                    binary=binary,
                    bbox=estimated_bbox,
                    image_width=width,
                    image_height=height,
                )
                if refined_bbox.w < self.OPTION_GROUP_MIN_WIDTH_RATIO:
                    continue
                if refined_bbox.w > self.OPTION_GROUP_MAX_WIDTH_RATIO:
                    continue
                boxes.append(refined_bbox)
        return deduplicate_bboxes(boxes, threshold=0.6)

    def _is_option_group_content(self, content: str) -> bool:
        """判断括号内容是否更像可选项而不是说明文字。"""

        normalized = " ".join((content or "").replace("\uFF0F", "/").split())
        if "/" not in normalized:
            return False
        slash_count = normalized.count("/")
        if slash_count < 1 or slash_count > self.OPTION_GROUP_MAX_SLASH_COUNT:
            return False
        if any(mark in normalized for mark in (":", "\uFF1A", ";", "\uFF1B", "?", "\uFF1F", "!", "\uFF01")):
            return False
        parts = [part.strip() for part in normalized.split("/") if part.strip()]
        if len(parts) < 2:
            return False
        text_units = self._measure_text_units(normalized)
        return self.OPTION_GROUP_MIN_TEXT_UNITS <= text_units <= self.OPTION_GROUP_MAX_TEXT_UNITS

    def _estimate_anchor_text_span_bbox(
        self,
        *,
        anchor_bbox: BBox,
        full_text: str,
        start_index: int,
        end_index: int,
        image_width: int,
        image_height: int,
    ) -> BBox:
        """根据锚点文本中的字符区间估算子串框。"""

        anchor_left, anchor_top, anchor_right, anchor_bottom = bbox_to_pixels(anchor_bbox, image_width, image_height)
        anchor_width = max(1, anchor_right - anchor_left)
        anchor_height = max(1, anchor_bottom - anchor_top)
        total_units = self._measure_text_units(full_text)
        start_units = self._measure_text_units(full_text[:start_index])
        end_units = self._measure_text_units(full_text[:end_index])

        estimated_left = anchor_left + int(anchor_width * (start_units / total_units))
        estimated_right = anchor_left + int(anchor_width * (end_units / total_units))
        padding_x = max(4, int(anchor_height * 0.35))
        padding_y = max(2, int(anchor_height * 0.20))

        estimated_left = max(anchor_left, estimated_left - padding_x)
        estimated_right = min(anchor_right, estimated_right + padding_x)
        estimated_top = max(0, anchor_top - padding_y)
        estimated_bottom = min(image_height, anchor_bottom + padding_y)

        if estimated_right <= estimated_left:
            estimated_right = min(image_width, estimated_left + max(8, anchor_height))
        if estimated_bottom <= estimated_top:
            estimated_bottom = min(image_height, estimated_top + max(8, anchor_height))

        return BBox(
            x=estimated_left / image_width,
            y=estimated_top / image_height,
            w=max(0.0, (estimated_right - estimated_left) / image_width),
            h=max(0.0, (estimated_bottom - estimated_top) / image_height),
        )

    def _refine_text_span_bbox(
        self,
        *,
        binary: np.ndarray,
        bbox: BBox,
        image_width: int,
        image_height: int,
    ) -> BBox:
        """利用二值墨迹收紧估算出的文本子串框。"""

        left, top, right, bottom = bbox_to_pixels(bbox, image_width, image_height)
        region = binary[top:bottom, left:right]
        if region.size == 0 or np.count_nonzero(region) == 0:
            return bbox

        rows = np.where(np.sum(region > 0, axis=1) > 0)[0]
        cols = np.where(np.sum(region > 0, axis=0) > 0)[0]
        if len(rows) == 0 or len(cols) == 0:
            return bbox

        padding = 2
        refined_left = max(0, left + int(cols[0]) - padding)
        refined_right = min(image_width, left + int(cols[-1]) + 1 + padding)
        refined_top = max(0, top + int(rows[0]) - padding)
        refined_bottom = min(image_height, top + int(rows[-1]) + 1 + padding)

        return BBox(
            x=refined_left / image_width,
            y=refined_top / image_height,
            w=max(0.0, (refined_right - refined_left) / image_width),
            h=max(0.0, (refined_bottom - refined_top) / image_height),
        )

    def _measure_text_units(self, text: str) -> float:
        """按近似显示宽度计算文本长度，用于从整行锚点切分子区域。"""

        total = 0.0
        for char in text:
            if char.isspace():
                total += 0.35
            elif ord(char) > 127:
                total += 1.8
            elif char in "[]{}()<>":
                total += 0.8
            elif char in "/\\|":
                total += 0.6
            elif char in ".,:;*'\"`":
                total += 0.45
            else:
                total += 1.0
        return max(total, 1.0)

    def _merge_date_like_line_groups(
        self,
        boxes: list[BBox],
        binary: np.ndarray,
        width: int,
        height: int,
        char_height_px: int,
    ) -> list[BBox]:
        """将被斜杠分开的日期短下划线合并为单个字段框。"""

        if not boxes:
            return []

        pixel_boxes = [bbox_to_pixels(bbox, width, height) for bbox in boxes]
        ordered_indices = sorted(range(len(pixel_boxes)), key=lambda index: (pixel_boxes[index][1], pixel_boxes[index][0]))
        used_indices: set[int] = set()
        merged_boxes: list[BBox] = []

        for index in ordered_indices:
            if index in used_indices:
                continue

            used_indices.add(index)
            current_left, current_top, current_right, current_bottom = pixel_boxes[index]

            while True:
                next_index = self._find_next_date_segment_index(
                    current_box=(current_left, current_top, current_right, current_bottom),
                    ordered_indices=ordered_indices,
                    pixel_boxes=pixel_boxes,
                    used_indices=used_indices,
                    binary=binary,
                    width=width,
                    height=height,
                    char_height_px=char_height_px,
                )
                if next_index is None:
                    break

                used_indices.add(next_index)
                next_left, next_top, next_right, next_bottom = pixel_boxes[next_index]
                current_left = min(current_left, next_left)
                current_top = min(current_top, next_top)
                current_right = max(current_right, next_right)
                current_bottom = max(current_bottom, next_bottom)

            merged_boxes.append(
                BBox(
                    x=current_left / width,
                    y=current_top / height,
                    w=max(0.0, (current_right - current_left) / width),
                    h=max(0.0, (current_bottom - current_top) / height),
                )
            )
        return merged_boxes

    def _find_next_date_segment_index(
        self,
        *,
        current_box: tuple[int, int, int, int],
        ordered_indices: list[int],
        pixel_boxes: list[tuple[int, int, int, int]],
        used_indices: set[int],
        binary: np.ndarray,
        width: int,
        height: int,
        char_height_px: int,
    ) -> int | None:
        """查找与当前日期段相邻且应合并的下一段下划线。"""

        current_left, current_top, current_right, current_bottom = current_box
        row_tolerance = max(4, int(char_height_px * 0.45))
        max_segment_width = max(24, int(char_height_px * self.DATE_SEGMENT_MAX_WIDTH_RATIO))

        for index in ordered_indices:
            if index in used_indices:
                continue

            candidate_left, candidate_top, candidate_right, candidate_bottom = pixel_boxes[index]
            candidate_width = candidate_right - candidate_left
            if candidate_left <= current_right:
                continue
            if candidate_width > max_segment_width:
                continue
            merged_width = candidate_right - current_left
            if merged_width > int(width * self.SEGMENT_GROUP_MAX_TOTAL_WIDTH_RATIO):
                continue
            if abs(candidate_top - current_top) > row_tolerance or abs(candidate_bottom - current_bottom) > row_tolerance:
                continue
            if not self._looks_like_date_separator(
                binary=binary,
                left_box=current_box,
                right_box=(candidate_left, candidate_top, candidate_right, candidate_bottom),
                image_height=height,
                char_height_px=char_height_px,
            ):
                continue
            return index
        return None

    def _looks_like_date_separator(
        self,
        *,
        binary: np.ndarray,
        left_box: tuple[int, int, int, int],
        right_box: tuple[int, int, int, int],
        image_height: int,
        char_height_px: int,
    ) -> bool:
        """判断两段下划线之间是否更像日期分隔符而不是普通单词。"""

        left_x1, left_y1, left_x2, left_y2 = left_box
        right_x1, right_y1, right_x2, right_y2 = right_box
        gap_width = right_x1 - left_x2
        if gap_width <= 0:
            return False
        if gap_width > max(6, int(char_height_px * self.DATE_SEGMENT_MAX_GAP_RATIO)):
            return False

        band_padding = max(2, char_height_px // 3)
        band_top = max(0, min(left_y1, right_y1) - band_padding)
        band_bottom = min(image_height, max(left_y2, right_y2) + band_padding)
        if band_bottom <= band_top:
            return False

        gap_region = binary[band_top:band_bottom, left_x2:right_x1]
        if gap_region.size == 0:
            return False

        ink_density = float(np.count_nonzero(gap_region)) / float(gap_region.size)
        if gap_width <= max(4, int(char_height_px * 0.55)) and ink_density < 0.01:
            return True
        if ink_density < 0.01 or ink_density > self.SEGMENT_SEPARATOR_MAX_DENSITY:
            return False

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(gap_region, connectivity=8)
        component_count = 0
        max_component_width = max(3, int(char_height_px * 0.95))
        max_component_height = max(4, int(char_height_px * 1.9))

        for label_index in range(1, num_labels):
            component_area = int(stats[label_index, cv2.CC_STAT_AREA])
            if component_area < 2:
                continue
            component_width = int(stats[label_index, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label_index, cv2.CC_STAT_HEIGHT])
            if component_width > max_component_width or component_height > max_component_height:
                return False
            component_count += 1
            if component_count > self.SEGMENT_SEPARATOR_MAX_COMPONENTS:
                return False

        return component_count >= 1

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
    CHECKBOX_MAX_INTERIOR_FILL = 0.15
    CHECKBOX_CHECKED_MAX_INTERIOR_FILL = 0.34
    CHECKBOX_SIDE_BAND_DIVISOR = 8
    CHECKBOX_SIDE_MIN_FILL = 0.12
    CHECKBOX_REQUIRED_SIDE_COUNT = 3

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
            if len(approx) < 4 or len(approx) > 10:
                continue

            # 内部留白：bbox 内缩 ~1/6 后的核心区域必须接近空白。
            # 对已勾选框，允许核心区出现勾线，但仍要求边框签名明显，避免放进正文字符。
            margin = max(2, min(w, h) // 6)
            interior = roi[margin:h - margin, margin:w - margin]
            if interior.size > 0:
                interior_fill = float(np.count_nonzero(interior)) / (interior.size + 1e-6)
                if interior_fill > self.CHECKBOX_MAX_INTERIOR_FILL:
                    if interior_fill > self.CHECKBOX_CHECKED_MAX_INTERIOR_FILL:
                        continue
                    if not self._has_checkbox_border_signature(roi):
                        continue
                elif not self._has_checkbox_border_signature(roi):
                    continue

            # 量化到 6 px 网格抑制近位重复轮廓（同一方框的内/外描边）。
            pos_key = (x // 6, y // 6)
            if pos_key in seen_positions:
                continue
            seen_positions.add(pos_key)

            boxes.append(BBox(x=x / width, y=y / height, w=w / width, h=h / height))
        return deduplicate_bboxes(boxes)

    def _has_checkbox_border_signature(self, roi: np.ndarray) -> bool:
        """判断候选区域是否具备方框边框特征。"""

        if roi.size == 0:
            return False

        height, width = roi.shape[:2]
        band = max(2, min(width, height) // self.CHECKBOX_SIDE_BAND_DIVISOR)
        if band * 2 >= width or band * 2 >= height:
            return False

        top_fill = self._compute_binary_fill_ratio(roi[:band, :])
        bottom_fill = self._compute_binary_fill_ratio(roi[height - band:, :])
        left_fill = self._compute_binary_fill_ratio(roi[:, :band])
        right_fill = self._compute_binary_fill_ratio(roi[:, width - band:])
        strong_side_count = sum(
            fill >= self.CHECKBOX_SIDE_MIN_FILL
            for fill in (top_fill, bottom_fill, left_fill, right_fill)
        )
        return strong_side_count >= self.CHECKBOX_REQUIRED_SIDE_COUNT

    def _compute_binary_fill_ratio(self, region: np.ndarray) -> float:
        """计算二值区域的着墨占比。"""

        if region.size == 0:
            return 0.0
        return float(np.count_nonzero(region)) / float(region.size + 1e-6)

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

    def _find_nearest_anchor(self, bbox: BBox, anchors: list[AnchorText]) -> AnchorText | None:
        """查找与字段最近的锚点对象。"""

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
        return best_anchor

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

    def _should_reject_non_field_line_candidate(
        self,
        *,
        bbox: BBox,
        anchors: list[AnchorText],
        nearest_anchor: AnchorText | None,
    ) -> bool:
        """根据顶部页眉和标题锚点关系过滤非字段横线。"""

        if bbox.w < self.HEADER_DECORATION_MIN_WIDTH_RATIO:
            return False
        if bbox.y > self.HEADER_DECORATION_MAX_Y_RATIO:
            return False
        if self._has_left_label_anchor(bbox, anchors):
            return False
        if nearest_anchor is None:
            return bbox.w >= 0.75

        anchor_bottom = nearest_anchor.bbox.y + nearest_anchor.bbox.h
        vertical_gap = bbox.y - anchor_bottom
        if vertical_gap < -0.01:
            return False
        max_vertical_gap = max(0.012, nearest_anchor.bbox.h * self.HEADER_ANCHOR_MAX_GAP_RATIO)
        if vertical_gap > max_vertical_gap:
            return False
        return self._horizontal_overlap_ratio(bbox, nearest_anchor.bbox) >= self.HEADER_ANCHOR_OVERLAP_RATIO

    def _is_text_embedded_short_line_candidate(
        self,
        *,
        bbox: BBox,
        anchors: list[AnchorText],
    ) -> bool:
        """判断短横线是否被 OCR 文本框包裹，更像文字横笔而不是填写下划线。"""

        if bbox.w > self.TEXT_EMBEDDED_SHORT_LINE_MAX_WIDTH_RATIO:
            return False
        if self._has_left_label_anchor(bbox, anchors):
            return False

        line_left = bbox.x
        line_right = bbox.x + bbox.w
        line_center_y = bbox.y + bbox.h / 2.0

        for anchor in anchors:
            anchor_left = anchor.bbox.x - self.TEXT_EMBEDDED_SHORT_LINE_ANCHOR_X_PADDING_RATIO
            anchor_right = anchor.bbox.x + anchor.bbox.w + self.TEXT_EMBEDDED_SHORT_LINE_ANCHOR_X_PADDING_RATIO
            if line_left < anchor_left or line_right > anchor_right:
                continue

            anchor_top = anchor.bbox.y - self.TEXT_EMBEDDED_SHORT_LINE_ANCHOR_TOP_PADDING_RATIO
            anchor_mid_bottom = anchor.bbox.y + anchor.bbox.h * self.TEXT_EMBEDDED_SHORT_LINE_MAX_CENTER_Y_RATIO
            anchor_bottom = anchor.bbox.y + anchor.bbox.h + max(
                self.TEXT_EMBEDDED_SHORT_LINE_ANCHOR_TOP_PADDING_RATIO,
                anchor.bbox.h * self.TEXT_EMBEDDED_SHORT_LINE_ANCHOR_BOTTOM_PADDING_RATIO,
            )
            if anchor_top <= line_center_y <= min(anchor_mid_bottom, anchor_bottom):
                return True
        return False

    def _has_left_label_anchor(self, bbox: BBox, anchors: list[AnchorText]) -> bool:
        """判断字段左侧是否存在同一行的标签锚点。"""

        field_center_y = bbox.y + bbox.h / 2.0
        vertical_tolerance = max(0.02, bbox.h * 2.5)
        max_horizontal_gap = max(0.03, bbox.h * 8.0)

        for anchor in anchors:
            anchor_right = anchor.bbox.x + anchor.bbox.w
            anchor_center_y = anchor.bbox.y + anchor.bbox.h / 2.0
            if anchor_right > bbox.x + 0.01:
                continue
            if abs(anchor_center_y - field_center_y) > vertical_tolerance:
                continue
            if bbox.x - anchor_right > max_horizontal_gap:
                continue
            return True
        return False

    def _horizontal_overlap_ratio(self, bbox: BBox, other_bbox: BBox) -> float:
        """计算另一个框在当前框水平方向上的覆盖比例。"""

        overlap_left = max(bbox.x, other_bbox.x)
        overlap_right = min(bbox.x + bbox.w, other_bbox.x + other_bbox.w)
        overlap_width = max(0.0, overlap_right - overlap_left)
        return overlap_width / max(bbox.w, 1e-6)

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
        rotation_degree: float = 0.0,
        line_anchor_text: str | None,
        canonical_key_id: str | None,
        key_hint: str | None,
        key_hint_en: str | None,
    ) -> FieldCandidate:
        """裁切字段并生成最终候选对象。"""

        display_bbox = bbox
        crop_bbox = expand_bbox_asymmetric(
            bbox,
            left_padding=self.FIELD_CROP_LEFT_RIGHT_PADDING,
            top_padding=self.FIELD_CROP_TOP_BOTTOM_PADDING,
            right_padding=self.FIELD_CROP_LEFT_RIGHT_PADDING,
            bottom_padding=self.FIELD_CROP_TOP_BOTTOM_PADDING,
        )

        crop_preprocessed = crop_by_bbox(preprocessed_image, crop_bbox)
        preprocessed_height, preprocessed_width = preprocessed_image.shape[:2]
        origin_crop_bbox = map_bbox_to_original_space(
            crop_bbox,
            preprocessed_width,
            preprocessed_height,
            rotation_degree,
        )
        crop_origin = crop_by_bbox(origin_image, origin_crop_bbox)
        crop_path = self.repository.build_artifact_path(task_id, "fields", f"{field_id}.png")
        cv2.imwrite(str(crop_path), crop_origin)

        ink_ratio = round(compute_ink_ratio(crop_preprocessed), 4)
        ocr_text = None
        ocr_confidence = None
        is_checked = None
        is_filled = ink_ratio >= self.settings.text_filled_ink_threshold

        if detector_source == "anchor_option_detector":
            ocr_text = None
            ocr_confidence = 0.0
            is_filled = False
        elif field_type in {FieldType.TEXT, FieldType.DATE, FieldType.HANDWRITING}:
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
