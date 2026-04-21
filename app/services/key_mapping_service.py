"""阶段 3 Key Mapping 实现。"""

from __future__ import annotations

import json

from app.core.config import Settings
from app.models.common import ConfidenceLevel, FieldType, KeySource, MatchStatus
from app.models.stages import (
    FieldCandidate,
    FusionOutput,
    KeyAliasDefinition,
    KeyMappingOutput,
    KeyedField,
    LeftChannelOutput,
    PendingKey,
    RouteDecision,
)
from app.repositories.key_alias_repository import KeyAliasRepository
from app.repositories.template_repository import TemplateRepository
from app.services.local_model_client import OpenAiCompatibleClient
from app.utils.text_utils import normalize_text


class KeyMappingService:
    """负责将融合字段映射到稳定 canonical key。"""

    ALIAS_MATCH_THRESHOLD = 0.75
    LLM_MATCH_THRESHOLD = 0.65

    def __init__(
        self,
        settings: Settings,
        template_repository: TemplateRepository,
        key_alias_repository: KeyAliasRepository,
        llm_client: OpenAiCompatibleClient,
    ) -> None:
        """注入依赖。"""

        self.settings = settings
        self.template_repository = template_repository
        self.key_alias_repository = key_alias_repository
        self.llm_client = llm_client

    def build_from_left_output(
        self,
        left_output: LeftChannelOutput,
        route_decision: RouteDecision,
    ) -> KeyMappingOutput:
        """将 M1 左通道结果转换为可导出的 Key Mapping 输出。"""

        template_fields = self._load_template_fields(route_decision)
        alias_lookup = self._load_alias_lookup()
        keyed_fields: list[KeyedField] = []

        for page in left_output.pages:
            for field in page.field_candidates:
                direct_key = self._build_keyed_field_from_left(field, template_fields, alias_lookup)
                keyed_fields.append(direct_key)

        return KeyMappingOutput(
            task_id=left_output.task_id,
            keyed_fields=keyed_fields,
            pending_keys=[],
        )

    def run(
        self,
        fusion_output: FusionOutput,
        route_decision: RouteDecision,
    ) -> KeyMappingOutput:
        """执行阶段 3 key mapping。"""

        template_fields = self._load_template_fields(route_decision)
        alias_lookup = self._load_alias_lookup()
        keyed_fields: list[KeyedField] = []
        pending_keys: list[PendingKey] = []

        for field in fusion_output.merged_fields:
            keyed_field, pending_key = self._map_fusion_field(field, template_fields, alias_lookup)
            keyed_fields.append(keyed_field)
            if pending_key is not None:
                pending_keys.append(pending_key)

        return KeyMappingOutput(
            task_id=fusion_output.task_id,
            form_title=fusion_output.form_title,
            form_type=fusion_output.form_type,
            keyed_fields=keyed_fields,
            pending_keys=pending_keys,
        )

    def _build_keyed_field_from_left(
        self,
        field: FieldCandidate,
        template_fields: dict[str, dict],
        alias_lookup: dict[str, KeyAliasDefinition],
    ) -> KeyedField:
        """构建 M1 路径使用的映射字段。"""

        if field.canonical_key_id and field.canonical_key_id in template_fields:
            template_field = template_fields[field.canonical_key_id]
            canonical_meta = alias_lookup.get(field.canonical_key_id)
            key = template_field.get("key") or (canonical_meta.canonical_key if canonical_meta else None) or field.key_hint or field.line_anchor_text or field.field_id
            key_en = template_field.get("key_en") or (canonical_meta.canonical_key_en if canonical_meta else None) or field.key_hint_en
            confidence = max(float(field.ocr_confidence or 0.0), 0.85)
            return KeyedField(
                field_id=field.field_id,
                task_id=field.task_id,
                page_index=field.page_index,
                field_type=field.field_type,
                canonical_key_id=field.canonical_key_id,
                canonical_key=key,
                canonical_key_en=key_en,
                key=key,
                key_en=key_en,
                key_raw=field.key_hint or field.line_anchor_text or field.field_id,
                key_source=KeySource.TEMPLATE_CANONICAL,
                value=self._resolve_left_value(field),
                value_source=self._resolve_left_value_source(field),
                confidence=confidence,
                confidence_level=self._to_confidence_level(confidence),
                bbox=field.bbox,
                crop_bbox=field.crop_bbox,
                crop_image_url=field.crop_image_url,
                is_filled=field.is_filled,
                review_required=False,
                mapping_reason="模板字段直接映射。",
                semantic_label=field.key_hint or field.line_anchor_text,
                match_status=MatchStatus.LEFT_ONLY,
            )

        key = field.key_hint or field.line_anchor_text or field.field_id
        confidence = max(float(field.ocr_confidence or 0.0), 0.0)
        return KeyedField(
            field_id=field.field_id,
            task_id=field.task_id,
            page_index=field.page_index,
            field_type=field.field_type,
            canonical_key_id=None,
            canonical_key=None,
            canonical_key_en=None,
            key=key,
            key_en=field.key_hint_en,
            key_raw=key,
            key_source=KeySource.NEW_FIELD,
            value=self._resolve_left_value(field),
            value_source=self._resolve_left_value_source(field),
            confidence=confidence,
            confidence_level=self._to_confidence_level(confidence),
            bbox=field.bbox,
            crop_bbox=field.crop_bbox,
            crop_image_url=field.crop_image_url,
            is_filled=field.is_filled,
            review_required=True,
            mapping_reason="M1 未启用语义映射，保留原始字段键。",
            semantic_label=field.key_hint or field.line_anchor_text,
            match_status=MatchStatus.LEFT_ONLY,
        )

    def _map_fusion_field(
        self,
        field,
        template_fields: dict[str, dict],
        alias_lookup: dict[str, KeyAliasDefinition],
    ) -> tuple[KeyedField, PendingKey | None]:
        """映射单个融合字段。"""

        if field.canonical_key_id and field.canonical_key_id in template_fields:
            template_field = template_fields[field.canonical_key_id]
            alias_meta = alias_lookup.get(field.canonical_key_id)
            key = template_field.get("key") or (alias_meta.canonical_key if alias_meta else None) or field.key_hint or field.key_raw
            key_en = template_field.get("key_en") or (alias_meta.canonical_key_en if alias_meta else None) or field.key_hint_en
            return (
                self._build_keyed_field(
                    field=field,
                    canonical_key_id=field.canonical_key_id,
                    canonical_key=key,
                    canonical_key_en=key_en,
                    key_source=KeySource.TEMPLATE_CANONICAL,
                    review_required=field.review_required,
                    mapping_reason="命中模板字段，直接复用 canonical key。",
                ),
                None,
            )

        alias_match = self._match_alias(field, alias_lookup)
        if alias_match is not None and alias_match["confidence"] >= self.ALIAS_MATCH_THRESHOLD:
            alias_def = alias_match["definition"]
            return (
                self._build_keyed_field(
                    field=field,
                    canonical_key_id=alias_def.canonical_key_id,
                    canonical_key=alias_def.canonical_key,
                    canonical_key_en=alias_def.canonical_key_en,
                    key_source=KeySource.TEMPLATE_CANONICAL,
                    review_required=field.review_required or alias_match["confidence"] < 0.85,
                    mapping_reason=f"命中本地别名表，匹配分数 {alias_match['confidence']:.2f}。",
                ),
                None,
            )

        llm_match = self._match_with_llm(field, alias_lookup)
        if llm_match is not None and llm_match["confidence"] >= self.LLM_MATCH_THRESHOLD:
            alias_def = llm_match["definition"]
            return (
                self._build_keyed_field(
                    field=field,
                    canonical_key_id=alias_def.canonical_key_id,
                    canonical_key=alias_def.canonical_key,
                    canonical_key_en=alias_def.canonical_key_en,
                    key_source=KeySource.TEMPLATE_CANONICAL,
                    review_required=True,
                    mapping_reason=llm_match["reason"],
                ),
                PendingKey(
                    field_id=field.field_id,
                    page_index=field.page_index,
                    field_type=field.field_type,
                    key_raw=field.key_raw,
                    suggested_canonical_key_name=alias_def.canonical_key,
                    reason="LLM 进行了低风险建议映射，仍需人工确认。",
                ),
            )

        key, key_en = self._split_label(field.semantic_label or field.key_raw)
        return (
            self._build_keyed_field(
                field=field,
                canonical_key_id=None,
                canonical_key=None,
                canonical_key_en=None,
                key_source=KeySource.VLM_RAW if field.semantic_label else KeySource.NEW_FIELD,
                review_required=True,
                mapping_reason="未找到稳定 canonical key，保留原始语义标签。",
                override_key=key,
                override_key_en=key_en,
            ),
            PendingKey(
                field_id=field.field_id,
                page_index=field.page_index,
                field_type=field.field_type,
                key_raw=field.key_raw,
                suggested_canonical_key_name=key,
                reason="别名表与本地 LLM 均未给出高置信映射。",
            ),
        )

    def _build_keyed_field(
        self,
        *,
        field,
        canonical_key_id: str | None,
        canonical_key: str | None,
        canonical_key_en: str | None,
        key_source: KeySource,
        review_required: bool,
        mapping_reason: str,
        override_key: str | None = None,
        override_key_en: str | None = None,
    ) -> KeyedField:
        """构建 KeyMapping 阶段字段对象。"""

        key = override_key or canonical_key or field.key_hint or field.line_anchor_text or field.key_raw or field.field_id
        key_en = override_key_en or canonical_key_en or field.key_hint_en
        return KeyedField(
            field_id=field.field_id,
            task_id=field.task_id,
            page_index=field.page_index,
            field_type=field.field_type,
            canonical_key_id=canonical_key_id,
            canonical_key=canonical_key,
            canonical_key_en=canonical_key_en,
            key=key,
            key_en=key_en,
            key_raw=field.key_raw,
            key_source=key_source,
            value=field.value,
            value_source=field.value_source,
            confidence=field.confidence,
            confidence_level=field.confidence_level,
            bbox=field.bbox,
            crop_bbox=field.crop_bbox,
            crop_image_url=field.crop_image_url,
            is_filled=field.is_filled,
            review_required=review_required,
            mapping_reason=mapping_reason,
            semantic_label=field.semantic_label,
            match_status=field.match_status,
            correction=field.correction,
        )

    def _resolve_left_value(self, field: FieldCandidate):
        """解析左通道字段值。"""

        field_type = self._normalize_field_type(field.field_type)
        if field_type == FieldType.CHECKBOX.value:
            return bool(field.is_checked)
        if field_type == FieldType.SIGNATURE.value:
            return bool(field.is_filled)
        return field.ocr_text

    def _resolve_left_value_source(self, field: FieldCandidate):
        """解析左通道值来源。"""

        field_type = self._normalize_field_type(field.field_type)
        if field_type in {FieldType.CHECKBOX.value, FieldType.SIGNATURE.value}:
            from app.models.common import ValueSource

            return ValueSource.CROP_ONLY
        from app.models.common import ValueSource

        return ValueSource.OCR

    def _normalize_field_type(self, value: FieldType | str) -> str:
        """将字段类型统一归一化为字符串值。"""

        if isinstance(value, FieldType):
            return value.value
        return str(value).strip().lower()

    def _load_template_fields(self, route_decision: RouteDecision) -> dict[str, dict]:
        """读取命中模板中的字段定义。"""

        if route_decision.matched_template is None:
            return {}

        registry = self.template_repository.load_registry()
        for template in registry.templates:
            if (
                template.template_id == route_decision.matched_template.template_id
                and template.template_revision == route_decision.matched_template.template_revision
            ):
                return {
                    field.canonical_key_id: {
                        "key": field.key,
                        "key_en": field.key_en,
                    }
                    for field in template.fields
                }
        return {}

    def _load_alias_lookup(self) -> dict[str, KeyAliasDefinition]:
        """读取字段别名表。"""

        registry = self.key_alias_repository.load_registry()
        return {
            item.canonical_key_id: item
            for item in registry.keys
        }

    def _match_alias(self, field, alias_lookup: dict[str, KeyAliasDefinition]) -> dict | None:
        """在本地别名表中查找最佳匹配。"""

        texts = self._candidate_texts(field)
        best_definition = None
        best_confidence = 0.0
        for alias_definition in alias_lookup.values():
            for candidate_text in texts:
                confidence = self._score_alias(candidate_text, alias_definition)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_definition = alias_definition
        if best_definition is None:
            return None
        return {
            "definition": best_definition,
            "confidence": best_confidence,
        }

    def _match_with_llm(self, field, alias_lookup: dict[str, KeyAliasDefinition]) -> dict | None:
        """使用本地 LLM 进行语义映射兜底。"""

        if not self.llm_client.configured or not self.settings.llm_model.strip() or not alias_lookup:
            return None

        candidate_payload = [
            {
                "canonical_key_id": definition.canonical_key_id,
                "canonical_key": definition.canonical_key,
                "canonical_key_en": definition.canonical_key_en,
                "aliases": definition.aliases,
            }
            for definition in alias_lookup.values()
        ]
        prompt = {
            "task": "根据字段语义为表单字段选择最合适的 canonical key。",
            "field": {
                "field_id": field.field_id,
                "field_type": field.field_type,
                "semantic_label": field.semantic_label,
                "key_raw": field.key_raw,
                "line_anchor_text": field.line_anchor_text,
                "value": field.value,
            },
            "candidates": candidate_payload,
            "output_schema": {
                "canonical_key_id": "string|null",
                "confidence": "number",
                "reason": "string",
            },
            "rules": [
                "只返回 JSON。",
                "如果没有把握，请返回 canonical_key_id=null。",
                "confidence 必须是 0 到 1 的数字。",
            ],
        }
        try:
            response = self.llm_client.chat_completions(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是表单字段语义映射助手，必须只返回合法 JSON。",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(prompt, ensure_ascii=False, indent=2),
                    },
                ],
                temperature=0.0,
            )
            content = self.llm_client.extract_message_text(response)
            payload = self.llm_client.extract_json_payload(content)
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None
        canonical_key_id = str(payload.get("canonical_key_id") or "").strip()
        if canonical_key_id not in alias_lookup:
            return None
        confidence = self._to_float(payload.get("confidence"))
        return {
            "definition": alias_lookup[canonical_key_id],
            "confidence": confidence,
            "reason": str(payload.get("reason") or "本地 LLM 给出了 canonical key 建议。").strip(),
        }

    def _candidate_texts(self, field) -> list[str]:
        """收集用于映射的候选文本。"""

        values = [
            field.semantic_label,
            field.key_raw,
            getattr(field, "line_anchor_text", None),
            getattr(field, "key_hint", None),
        ]
        return [value for value in values if isinstance(value, str) and value.strip()]

    def _score_alias(self, text: str, alias_definition: KeyAliasDefinition) -> float:
        """计算文本与别名定义的匹配分数。"""

        normalized_text = normalize_text(text)
        if not normalized_text:
            return 0.0
        candidate_values = [
            alias_definition.canonical_key,
            alias_definition.canonical_key_en,
            *alias_definition.aliases,
        ]
        best_score = 0.0
        for candidate_value in candidate_values:
            normalized_candidate = normalize_text(candidate_value or "")
            if not normalized_candidate:
                continue
            if normalized_text == normalized_candidate:
                return 1.0
            if normalized_text in normalized_candidate or normalized_candidate in normalized_text:
                best_score = max(best_score, 0.82)
            else:
                overlap = self._token_overlap(normalized_text, normalized_candidate)
                best_score = max(best_score, overlap)
        return best_score

    def _token_overlap(self, left_text: str, right_text: str) -> float:
        """计算两个归一化文本的 token 重合度。"""

        left_tokens = set(self._split_tokens(left_text))
        right_tokens = set(self._split_tokens(right_text))
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _split_tokens(self, text: str) -> list[str]:
        """将文本拆成粗粒度 token。"""

        if len(text) <= 4:
            return [text]
        return [text[index:index + 2] for index in range(0, len(text) - 1)]

    def _split_label(self, label: str) -> tuple[str, str | None]:
        """拆分中英文混合标签。"""

        text = (label or "").strip()
        if " / " in text:
            left, right = text.split(" / ", 1)
            return left.strip() or text, right.strip() or None
        if "/" in text:
            left, right = text.split("/", 1)
            return left.strip() or text, right.strip() or None
        return text or "未命名字段", None

    def _to_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """将数值置信度转为等级。"""

        if confidence >= 0.90:
            return ConfidenceLevel.HIGH
        if confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _to_float(self, value: object) -> float:
        """将任意值安全转换为浮点数。"""

        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        return max(0.0, min(1.0, number))
