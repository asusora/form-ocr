"""阶段 4：结构化导出实现。"""

from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.models.common import ConfidenceLevel, FieldType, KeySource, TaskContext, ValueSource
from app.models.stages import (
    ExportMetadata,
    ExportOutput,
    ExportPage,
    ExportedField,
    KvPairEntry,
    LeftChannelOutput,
    RouteDecision,
    Stage0Output,
)
from app.repositories.task_repository import TaskRepository
from app.utils.image_utils import map_bbox_to_original_space
from app.utils.text_utils import build_kv_key, make_raw_key


class ExportService:
    """负责将 M1 结果导出为统一 JSON。"""

    def __init__(self, settings: Settings, repository: TaskRepository) -> None:
        """注入依赖。"""

        self.settings = settings
        self.repository = repository

    def run(
        self,
        task_context: TaskContext,
        stage0_output: Stage0Output,
        route_decision: RouteDecision,
        left_output: LeftChannelOutput,
        *,
        processing_time_ms: int,
    ) -> ExportOutput:
        """生成阶段 4 导出对象。"""

        exported_fields: list[ExportedField] = []
        kv_pairs: dict[str, KvPairEntry] = {}
        used_kv_keys: set[str] = set()
        keys_from_template = 0
        page_lookup = {page.page_index: page for page in stage0_output.pages}

        for page in left_output.pages:
            for field in page.field_candidates:
                page_asset = page_lookup.get(field.page_index)
                key_raw = make_raw_key(field.line_anchor_text or field.key_hint or "", field.field_id)
                key = field.key_hint or field.line_anchor_text or field.field_id
                key_source = (
                    KeySource.TEMPLATE_CANONICAL
                    if field.canonical_key_id
                    else KeySource.NEW_FIELD
                )
                if key_source == KeySource.TEMPLATE_CANONICAL:
                    keys_from_template += 1
                value, value_source, confidence = self._resolve_field_value(field)
                confidence_level = self._to_confidence_level(confidence)
                review_required = confidence_level == ConfidenceLevel.LOW or key == field.field_id
                dirty_reason = (
                    "日期格式疑似异常"
                    if field.field_type == FieldType.DATE and value and not self._is_date_like(str(value))
                    else None
                )
                if dirty_reason:
                    review_required = True
                origin_bbox = (
                    map_bbox_to_original_space(
                        field.bbox,
                        page_asset.width,
                        page_asset.height,
                        page_asset.rotation_degree,
                    )
                    if page_asset is not None
                    else field.bbox
                )

                exported_field = ExportedField(
                    field_id=field.field_id,
                    page_index=field.page_index,
                    canonical_key_id=field.canonical_key_id,
                    key=key,
                    key_en=field.key_hint_en,
                    key_source=key_source,
                    key_raw=key_raw,
                    value=value,
                    value_source=value_source,
                    confidence=confidence,
                    confidence_level=confidence_level,
                    bbox=field.bbox,
                    origin_bbox=origin_bbox,
                    crop_bbox=field.crop_bbox,
                    crop_image_url=field.crop_image_url,
                    field_type=field.field_type,
                    is_filled=field.is_filled,
                    dirty=bool(dirty_reason),
                    dirty_reason=dirty_reason,
                    review_required=review_required,
                    user_modified=False,
                    user_confirmed=False,
                )
                exported_fields.append(exported_field)

                kv_key = self._build_unique_kv_key(
                    canonical_key_id=field.canonical_key_id,
                    key_raw=key_raw,
                    fallback=field.field_id,
                    used_keys=used_kv_keys,
                )
                display_name = key if not field.key_hint_en else f"{key} / {field.key_hint_en}"
                kv_pairs[kv_key] = KvPairEntry(
                    display_name=display_name,
                    value=value,
                    field_type=field.field_type,
                    page_index=field.page_index,
                    field_id=field.field_id,
                )

        export_output = ExportOutput(
            task_id=task_context.task_id,
            template_id=route_decision.matched_template.template_id if route_decision.matched_template else None,
            template_revision=route_decision.matched_template.template_revision if route_decision.matched_template else None,
            template_match_confidence=route_decision.matched_template.confidence if route_decision.matched_template else None,
            pages=[
                ExportPage(
                    page_index=page.page_index,
                    page_image_url=page.origin_image_url,
                    page_preprocessed_image_url=page.preprocessed_image_url,
                )
                for page in stage0_output.pages
            ],
            fields=exported_fields,
            kv_pairs=kv_pairs,
            metadata=ExportMetadata(
                processing_time_ms=processing_time_ms,
                ocr_engine=self.settings.ocr_engine,
                vlm_model=self.settings.default_vlm_model,
                template_matched=route_decision.matched_template is not None,
                artifact_version=task_context.artifact_version,
                keys_from_template=keys_from_template,
                keys_from_vlm_raw=0,
                corrections_made=0,
            ),
        )
        self._save_export_json(task_context.task_id, export_output)
        return export_output

    def _save_export_json(self, task_id: str, export_output: ExportOutput) -> None:
        """写入导出 JSON 工件。"""

        export_path = self.repository.build_artifact_path(task_id, "results", "export.json")
        export_path.write_text(
            json.dumps(export_output.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_unique_kv_key(
        self,
        *,
        canonical_key_id: str | None,
        key_raw: str,
        fallback: str,
        used_keys: set[str],
    ) -> str:
        """生成唯一的 KV 键，避免重复字段覆盖。"""

        if canonical_key_id:
            candidate = canonical_key_id
            suffix = 2
            while candidate in used_keys:
                candidate = f"{canonical_key_id}__{suffix}"
                suffix += 1
            used_keys.add(candidate)
            return candidate
        return build_kv_key(key_raw, fallback, used_keys)

    def _resolve_field_value(self, field) -> tuple[Any, ValueSource, float]:
        """计算字段值、值来源和置信度。"""

        if field.field_type == FieldType.CHECKBOX:
            confidence = round(0.5 + min(field.ink_ratio, 0.5), 4)
            return bool(field.is_checked), ValueSource.CROP_ONLY, confidence
        if field.field_type == FieldType.SIGNATURE:
            confidence = round(max(0.35, min(field.ink_ratio + 0.4, 0.98)), 4)
            return bool(field.is_filled), ValueSource.CROP_ONLY, confidence
        confidence = round(max(field.ocr_confidence or 0.0, 0.0), 4)
        return field.ocr_text, ValueSource.OCR, confidence

    def _to_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """将数值置信度转为等级。"""

        if confidence >= 0.90:
            return ConfidenceLevel.HIGH
        if confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _is_date_like(self, value: str) -> bool:
        """做宽松日期校验。"""

        normalized = value.strip()
        if not normalized:
            return False
        for separator in ("-", "/", ".", "年"):
            if separator in normalized:
                return True
        digits = "".join(char for char in normalized if char.isdigit())
        return len(digits) in {6, 8}
