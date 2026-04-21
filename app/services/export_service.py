"""阶段 4：结构化导出实现。"""

from __future__ import annotations

from app.core.config import Settings
from app.models.common import ConfidenceLevel, FieldType, KeySource, TaskContext
from app.models.stages import (
    ExportMetadata,
    ExportOutput,
    ExportPage,
    ExportedField,
    KeyMappingOutput,
    KvPairEntry,
    RouteDecision,
    Stage0Output,
)
from app.repositories.task_repository import TaskRepository
from app.utils.image_utils import map_bbox_to_original_space
from app.utils.text_utils import build_kv_key


class ExportService:
    """负责将识别结果导出为统一 JSON。"""

    def __init__(self, settings: Settings, repository: TaskRepository) -> None:
        """注入依赖。"""

        self.settings = settings
        self.repository = repository

    def run(
        self,
        task_context: TaskContext,
        stage0_output: Stage0Output,
        route_decision: RouteDecision,
        keyed_output: KeyMappingOutput,
        *,
        processing_time_ms: int,
    ) -> ExportOutput:
        """生成阶段 4 导出对象。"""

        exported_fields: list[ExportedField] = []
        kv_pairs: dict[str, KvPairEntry] = {}
        used_kv_keys: set[str] = set()
        keys_from_template = 0
        keys_from_vlm_raw = 0
        corrections_made = 0
        page_lookup = {page.page_index: page for page in stage0_output.pages}

        for field in keyed_output.keyed_fields:
            page_asset = page_lookup.get(field.page_index)
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
            dirty_reason = (
                "日期格式疑似异常"
                if field.field_type == FieldType.DATE and field.value and not self._is_date_like(str(field.value))
                else field.dirty_reason
            )
            review_required = field.review_required or bool(dirty_reason)

            exported_field = ExportedField(
                field_id=field.field_id,
                page_index=field.page_index,
                canonical_key_id=field.canonical_key_id,
                key=field.key,
                key_en=field.key_en,
                key_source=field.key_source,
                key_raw=field.key_raw,
                value=field.value,
                value_source=field.value_source,
                confidence=field.confidence,
                confidence_level=field.confidence_level,
                bbox=field.bbox,
                origin_bbox=origin_bbox,
                crop_bbox=field.crop_bbox,
                crop_image_url=field.crop_image_url,
                field_type=field.field_type,
                is_filled=field.is_filled,
                dirty=bool(dirty_reason),
                dirty_reason=dirty_reason,
                review_required=review_required,
                user_modified=field.user_modified,
                user_confirmed=field.user_confirmed,
                semantic_label=field.semantic_label,
                match_status=field.match_status,
                mapping_reason=field.mapping_reason,
                correction=field.correction,
            )
            exported_fields.append(exported_field)

            if field.key_source == KeySource.TEMPLATE_CANONICAL:
                keys_from_template += 1
            if field.key_source == KeySource.VLM_RAW:
                keys_from_vlm_raw += 1
            if field.correction is not None:
                corrections_made += 1

            kv_key = self._build_unique_kv_key(
                canonical_key_id=field.canonical_key_id,
                key_raw=field.key_raw,
                fallback=field.field_id,
                used_keys=used_kv_keys,
            )
            display_name = field.key if not field.key_en else f"{field.key} / {field.key_en}"
            kv_pairs[kv_key] = KvPairEntry(
                display_name=display_name,
                value=field.value,
                field_type=field.field_type,
                page_index=field.page_index,
                field_id=field.field_id,
            )

        export_output = ExportOutput(
            task_id=task_context.task_id,
            template_id=route_decision.matched_template.template_id if route_decision.matched_template else None,
            template_revision=route_decision.matched_template.template_revision if route_decision.matched_template else None,
            template_match_confidence=route_decision.matched_template.confidence if route_decision.matched_template else None,
            form_title=keyed_output.form_title,
            form_type=keyed_output.form_type,
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
                vlm_model=self.settings.vlm_model,
                template_matched=route_decision.matched_template is not None,
                artifact_version=task_context.artifact_version,
                keys_from_template=keys_from_template,
                keys_from_vlm_raw=keys_from_vlm_raw,
                corrections_made=corrections_made,
            ),
        )
        self.repository.save_result_output(
            task_context.task_id,
            "export",
            export_output,
            artifact_version=task_context.artifact_version,
        )
        return export_output

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

    def _to_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """将数值置信度转为等级。"""

        if confidence >= 0.90:
            return ConfidenceLevel.HIGH
        if confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
