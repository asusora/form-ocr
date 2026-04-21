"""阶段 2 融合实现。"""

from __future__ import annotations

from typing import Any

from app.models.common import ConfidenceLevel, FieldType, KeySource, MatchStatus, ValueSource
from app.models.stages import (
    FieldCorrection,
    FusionOutput,
    LeftChannelOutput,
    MergedField,
    RightChannelOutput,
    RouteDecision,
    SemanticFieldCandidate,
    UnresolvedCandidate,
)
from app.utils.text_utils import normalize_text


class FusionService:
    """负责左右通道配对、冲突解决与字段融合。"""

    HANDWRITING_CONFIDENCE_THRESHOLD = 0.60
    MATCH_SCORE_THRESHOLD = 0.55

    def run(
        self,
        left_output: LeftChannelOutput,
        right_output: RightChannelOutput,
        route_decision: RouteDecision,
    ) -> FusionOutput:
        """执行阶段 2 融合。"""

        del route_decision

        left_pages = {page.page_index: page.field_candidates for page in left_output.pages}
        right_pages = {page.page_index: page.semantic_candidates for page in right_output.pages}
        page_indexes = sorted(set(left_pages.keys()) | set(right_pages.keys()))

        merged_fields: list[MergedField] = []
        unresolved_candidates: list[UnresolvedCandidate] = []

        for page_index in page_indexes:
            page_left_fields = left_pages.get(page_index, [])
            page_right_candidates = right_pages.get(page_index, [])
            page_merged, page_unresolved = self._merge_page(
                page_left_fields=page_left_fields,
                page_right_candidates=page_right_candidates,
            )
            merged_fields.extend(page_merged)
            unresolved_candidates.extend(page_unresolved)

        return FusionOutput(
            task_id=left_output.task_id,
            form_title=right_output.form_title,
            form_type=right_output.form_type,
            merged_fields=merged_fields,
            unresolved_candidates=unresolved_candidates,
        )

    def _merge_page(
        self,
        *,
        page_left_fields: list,
        page_right_candidates: list[SemanticFieldCandidate],
    ) -> tuple[list[MergedField], list[UnresolvedCandidate]]:
        """融合单页字段。"""

        left_by_id = {field.field_id: field for field in page_left_fields}
        assignments: dict[str, SemanticFieldCandidate] = {}
        used_left_ids: set[str] = set()
        used_right_ids: set[str] = set()

        sorted_right_candidates = sorted(
            page_right_candidates,
            key=lambda item: item.semantic_confidence,
            reverse=True,
        )

        for candidate in sorted_right_candidates:
            matched_field_id = None
            if candidate.field_id_hint and candidate.field_id_hint in left_by_id and candidate.field_id_hint not in used_left_ids:
                matched_field_id = candidate.field_id_hint
            else:
                best_score = 0.0
                for field in page_left_fields:
                    if field.field_id in used_left_ids:
                        continue
                    score = self._score_pair(field, candidate)
                    if score > best_score:
                        best_score = score
                        matched_field_id = field.field_id
                if best_score < self.MATCH_SCORE_THRESHOLD:
                    matched_field_id = None

            if matched_field_id is None:
                continue

            assignments[matched_field_id] = candidate
            used_left_ids.add(matched_field_id)
            used_right_ids.add(candidate.semantic_candidate_id)

        merged_fields: list[MergedField] = []
        for field in page_left_fields:
            candidate = assignments.get(field.field_id)
            merged_fields.append(self._build_merged_field(field, candidate))

        unresolved_candidates = [
            UnresolvedCandidate(
                semantic_candidate_id=candidate.semantic_candidate_id,
                page_index=candidate.page_index,
                field_type=candidate.field_type,
                semantic_label=candidate.semantic_label,
                reason="右通道发现语义字段，但左通道未找到稳定候选框",
                resolution="manual_review_candidate",
            )
            for candidate in sorted_right_candidates
            if candidate.semantic_candidate_id not in used_right_ids
        ]
        return merged_fields, unresolved_candidates

    def _score_pair(self, field, candidate: SemanticFieldCandidate) -> float:
        """计算左字段与右通道候选的匹配分数。"""

        score = 0.0
        if self._is_field_type_compatible(field.field_type, candidate.field_type):
            score += 0.30
        else:
            score -= 0.30

        score += 0.40 * self._text_similarity(field.line_anchor_text or field.key_hint, candidate.near_anchor_text)
        score += 0.20 * self._text_similarity(field.ocr_text, candidate.value_read)
        if candidate.region_hint and candidate.region_hint.anchor_text:
            score += 0.10 * self._text_similarity(field.line_anchor_text or field.key_hint, candidate.region_hint.anchor_text)
        score += 0.10 * candidate.semantic_confidence
        return max(0.0, min(1.0, score))

    def _build_merged_field(self, field, candidate: SemanticFieldCandidate | None) -> MergedField:
        """构建融合字段对象。"""

        if candidate is None:
            value, value_source = self._resolve_left_only_value(field)
            confidence = self._base_field_confidence(field)
            return MergedField(
                field_id=field.field_id,
                task_id=field.task_id,
                page_index=field.page_index,
                field_type=field.field_type,
                bbox=field.bbox,
                crop_bbox=field.crop_bbox,
                crop_image_url=field.crop_image_url,
                canonical_key_id=field.canonical_key_id,
                key_hint=field.key_hint,
                key_hint_en=field.key_hint_en,
                line_anchor_text=field.line_anchor_text,
                semantic_label=field.key_hint or field.line_anchor_text,
                key_raw=(field.key_hint or field.line_anchor_text or field.field_id),
                value=value,
                key_source=KeySource.TEMPLATE_CANONICAL if field.canonical_key_id else KeySource.NEW_FIELD,
                value_source=value_source,
                confidence=confidence,
                confidence_level=self._to_confidence_level(confidence),
                match_status=MatchStatus.LEFT_ONLY,
                review_required=not field.canonical_key_id or confidence < 0.70,
                ocr_text=field.ocr_text,
                ocr_confidence=field.ocr_confidence,
                is_filled=field.is_filled,
                ink_ratio=field.ink_ratio,
                is_checked=field.is_checked,
            )

        match_status = MatchStatus.PAIRED
        review_required = False
        if not self._is_field_type_compatible(field.field_type, candidate.field_type):
            match_status = MatchStatus.CONFLICTED
            review_required = True

        value, value_source, correction = self._resolve_paired_value(field, candidate, match_status)
        key_source = KeySource.TEMPLATE_CANONICAL if field.canonical_key_id else KeySource.VLM_RAW
        key_raw = candidate.semantic_label or field.key_hint or field.line_anchor_text or field.field_id
        confidence = self._combined_confidence(field, candidate)
        if match_status == MatchStatus.CONFLICTED:
            confidence = min(confidence, 0.69)
        if confidence < 0.70:
            review_required = True

        return MergedField(
            field_id=field.field_id,
            task_id=field.task_id,
            page_index=field.page_index,
            field_type=field.field_type,
            bbox=field.bbox,
            crop_bbox=field.crop_bbox,
            crop_image_url=field.crop_image_url,
            canonical_key_id=field.canonical_key_id,
            key_hint=field.key_hint,
            key_hint_en=field.key_hint_en,
            line_anchor_text=field.line_anchor_text,
            semantic_label=candidate.semantic_label,
            key_raw=key_raw,
            value=value,
            key_source=key_source,
            value_source=value_source,
            confidence=confidence,
            confidence_level=self._to_confidence_level(confidence),
            match_status=match_status,
            review_required=review_required,
            correction=correction,
            ocr_text=field.ocr_text,
            ocr_confidence=field.ocr_confidence,
            is_filled=field.is_filled,
            ink_ratio=field.ink_ratio,
            is_checked=field.is_checked,
        )

    def _resolve_left_only_value(self, field) -> tuple[Any, ValueSource]:
        """在没有右通道结果时确定字段值。"""

        if field.field_type == FieldType.CHECKBOX:
            return bool(field.is_checked), ValueSource.CROP_ONLY
        if field.field_type == FieldType.SIGNATURE:
            return bool(field.is_filled), ValueSource.CROP_ONLY
        return field.ocr_text, ValueSource.OCR

    def _resolve_paired_value(
        self,
        field,
        candidate: SemanticFieldCandidate,
        match_status: MatchStatus,
    ) -> tuple[Any, ValueSource, FieldCorrection | None]:
        """根据融合规则生成最终值。"""

        if field.field_type == FieldType.CHECKBOX:
            return bool(field.is_checked), ValueSource.CROP_ONLY, None
        if field.field_type == FieldType.SIGNATURE:
            return bool(field.is_filled), ValueSource.CROP_ONLY, None

        ocr_value = field.ocr_text
        vlm_value = candidate.value_read
        if not vlm_value:
            return ocr_value, ValueSource.OCR, None

        should_use_vlm = (
            field.field_type == FieldType.HANDWRITING
            or not (ocr_value or "").strip()
            or (field.ocr_confidence or 0.0) < self.HANDWRITING_CONFIDENCE_THRESHOLD
            or match_status == MatchStatus.CONFLICTED
        )
        if not should_use_vlm:
            return ocr_value, ValueSource.OCR, None

        correction = None
        value_source = ValueSource.VLM
        if normalize_text(ocr_value or "") != normalize_text(vlm_value):
            correction = FieldCorrection(
                ocr_original=ocr_value,
                vlm_original=vlm_value,
                final_value=vlm_value,
                reason="右通道在低置信或手写场景下提供了更可信的值。",
            )
            value_source = ValueSource.VLM_CORRECTED
        return vlm_value, value_source, correction

    def _base_field_confidence(self, field) -> float:
        """计算左通道单独结果的置信度。"""

        if field.field_type == FieldType.CHECKBOX:
            return round(0.55 + min(field.ink_ratio, 0.35), 4)
        if field.field_type == FieldType.SIGNATURE:
            return round(max(0.35, min(field.ink_ratio + 0.40, 0.95)), 4)
        return round(max(float(field.ocr_confidence or 0.0), 0.0), 4)

    def _combined_confidence(self, field, candidate: SemanticFieldCandidate) -> float:
        """融合左右通道的置信度。"""

        left_confidence = self._base_field_confidence(field)
        if field.field_type in {FieldType.CHECKBOX, FieldType.SIGNATURE}:
            return round(max(left_confidence, candidate.semantic_confidence * 0.8), 4)
        combined = (left_confidence * 0.55) + (candidate.semantic_confidence * 0.45)
        return round(max(0.0, min(1.0, combined)), 4)

    def _to_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """将数值置信度转为等级。"""

        if confidence >= 0.90:
            return ConfidenceLevel.HIGH
        if confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _text_similarity(self, left_text: str | None, right_text: str | None) -> float:
        """计算两个文本的近似度。"""

        left_normalized = normalize_text(left_text or "")
        right_normalized = normalize_text(right_text or "")
        if not left_normalized or not right_normalized:
            return 0.0
        if left_normalized == right_normalized:
            return 1.0
        if left_normalized in right_normalized or right_normalized in left_normalized:
            return 0.8
        left_tokens = set(self._split_tokens(left_normalized))
        right_tokens = set(self._split_tokens(right_normalized))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _split_tokens(self, text: str) -> list[str]:
        """将归一化文本粗略拆分为 token。"""

        if len(text) <= 4:
            return [text]
        return [text[index:index + 2] for index in range(0, len(text) - 1)]

    def _is_field_type_compatible(self, left_type: FieldType, right_type: FieldType) -> bool:
        """判断左右通道字段类型是否兼容。"""

        if left_type == right_type:
            return True
        compatible_pairs = {
            (FieldType.TEXT, FieldType.HANDWRITING),
            (FieldType.HANDWRITING, FieldType.TEXT),
        }
        return (left_type, right_type) in compatible_pairs
