"""阶段 1 右通道实现。"""

from __future__ import annotations

import json

from app.core.config import Settings
from app.models.common import FieldType
from app.models.stages import (
    LeftChannelOutput,
    RightChannelOutput,
    RightChannelPageOutput,
    RouteDecision,
    SemanticFieldCandidate,
    Stage0Output,
    RegionHint,
)
from app.repositories.task_repository import TaskRepository
from app.services.local_model_client import OpenAiCompatibleClient, build_local_file_url
from app.utils.id_utils import generate_semantic_candidate_id


class RightChannelService:
    """负责调用本地 VLM 生成字段语义候选。"""

    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        vlm_client: OpenAiCompatibleClient,
    ) -> None:
        """注入依赖。"""

        self.settings = settings
        self.repository = repository
        self.vlm_client = vlm_client

    def run(
        self,
        stage0_output: Stage0Output,
        route_decision: RouteDecision,
        left_output: LeftChannelOutput,
    ) -> RightChannelOutput:
        """执行右通道语义分析。"""

        left_fields_by_page = {
            page.page_index: page.field_candidates
            for page in left_output.pages
        }
        page_results: list[RightChannelPageOutput] = []

        if not self.vlm_client.configured or not self.settings.vlm_model.strip():
            for page in stage0_output.pages:
                page_results.append(RightChannelPageOutput(page_index=page.page_index))
            return RightChannelOutput(task_id=stage0_output.task_id, pages=page_results)

        for page in stage0_output.pages:
            left_fields = left_fields_by_page.get(page.page_index, [])
            try:
                page_payload = self._analyze_page(
                    stage0_output=stage0_output,
                    route_decision=route_decision,
                    page_index=page.page_index,
                    image_url=page.origin_image_url,
                    left_fields=left_fields,
                )
                page_results.append(page_payload)
            except Exception:
                page_results.append(RightChannelPageOutput(page_index=page.page_index))

        return RightChannelOutput(
            task_id=stage0_output.task_id,
            form_title=next((page.form_title for page in page_results if page.form_title), None),
            form_type=next((page.form_type for page in page_results if page.form_type), None),
            pages=page_results,
        )

    def _analyze_page(
        self,
        *,
        stage0_output: Stage0Output,
        route_decision: RouteDecision,
        page_index: int,
        image_url: str,
        left_fields: list,
    ) -> RightChannelPageOutput:
        """分析单页表单并解析语义结果。"""

        image_path = self.repository.url_to_path(image_url)
        file_url = build_local_file_url(image_path)
        messages = self._build_messages(
            task_id=stage0_output.task_id,
            page_index=page_index,
            route_decision=route_decision,
            left_fields=left_fields,
            file_url=file_url,
        )
        response = self.vlm_client.chat_completions(
            model=self.settings.vlm_model,
            messages=messages,
            temperature=0.0,
        )
        response_text = self.vlm_client.extract_message_text(response)
        response_json = self.vlm_client.extract_json_payload(response_text)
        return self._parse_page_payload(
            task_id=stage0_output.task_id,
            page_index=page_index,
            payload=response_json,
            left_fields=left_fields,
        )

    def _build_messages(
        self,
        *,
        task_id: str,
        page_index: int,
        route_decision: RouteDecision,
        left_fields: list,
        file_url: str,
    ) -> list[dict]:
        """构造 VLM 提示词。"""

        left_field_payload = []
        for field in left_fields:
            left_field_payload.append(
                {
                    "field_id": field.field_id,
                    "field_type": field.field_type,
                    "ocr_text": field.ocr_text,
                    "ocr_confidence": field.ocr_confidence,
                    "line_anchor_text": field.line_anchor_text,
                    "key_hint": field.key_hint,
                    "key_hint_en": field.key_hint_en,
                    "bbox": field.bbox.model_dump(mode="json"),
                }
            )

        prompt = {
            "task_id": task_id,
            "page_index": page_index,
            "route_type": route_decision.route_type,
            "matched_template": (
                route_decision.matched_template.model_dump(mode="json")
                if route_decision.matched_template is not None
                else None
            ),
            "left_channel_candidates": left_field_payload,
            "instructions": [
                "你是表单 OCR 右通道语义分析器。",
                "请结合整页图片与左通道候选，输出严格 JSON，不允许输出任何解释性文字。",
                "字段语义请尽量使用“中文 / English”双语格式。",
                "如果能确定语义对应的左通道字段，请填写 field_id_hint。",
                "只返回以下结构：form_title、form_type、semantic_candidates。",
                "semantic_candidates 中的每个对象必须包含 field_type、semantic_label、semantic_confidence。",
                "无法确定的字段不要臆造。",
            ],
            "json_schema": {
                "form_title": "string|null",
                "form_type": "string|null",
                "semantic_candidates": [
                    {
                        "field_id_hint": "string|null",
                        "field_type": "text|date|checkbox|signature|handwriting",
                        "semantic_label": "string",
                        "value_read": "string|null",
                        "is_signed": "boolean|null",
                        "near_anchor_text": "string|null",
                        "spatial_description": "string|null",
                        "region_hint": {
                            "anchor_text": "string|null",
                            "direction": "string|null",
                            "priority_zone": "string|null",
                        },
                        "semantic_confidence": "number",
                    }
                ],
            },
        }
        return [
            {
                "role": "system",
                "content": "你是一个专业的图像分析助手，必须只返回合法 JSON。",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(prompt, ensure_ascii=False, indent=2),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": file_url},
                    },
                ],
            },
        ]

    def _parse_page_payload(
        self,
        *,
        task_id: str,
        page_index: int,
        payload: dict,
        left_fields: list,
    ) -> RightChannelPageOutput:
        """将模型 JSON 解析为右通道输出对象。"""

        valid_field_ids = {field.field_id for field in left_fields}
        semantic_candidates: list[SemanticFieldCandidate] = []
        raw_candidates = payload.get("semantic_candidates") if isinstance(payload, dict) else []
        if not isinstance(raw_candidates, list):
            raw_candidates = []

        for candidate_index, item in enumerate(raw_candidates):
            if not isinstance(item, dict):
                continue
            semantic_label = str(item.get("semantic_label") or "").strip()
            if not semantic_label:
                continue
            field_id_hint = str(item.get("field_id_hint") or "").strip() or None
            if field_id_hint not in valid_field_ids:
                field_id_hint = None

            semantic_candidates.append(
                SemanticFieldCandidate(
                    semantic_candidate_id=generate_semantic_candidate_id(task_id, page_index, candidate_index),
                    task_id=task_id,
                    page_index=page_index,
                    semantic_label=semantic_label,
                    field_type=self._parse_field_type(item.get("field_type")),
                    value_read=self._to_optional_string(item.get("value_read")),
                    is_signed=self._to_optional_bool(item.get("is_signed")),
                    near_anchor_text=self._to_optional_string(item.get("near_anchor_text")),
                    spatial_description=self._to_optional_string(item.get("spatial_description")),
                    region_hint=self._parse_region_hint(item.get("region_hint")),
                    semantic_confidence=self._to_confidence(item.get("semantic_confidence")),
                    model_name=self.settings.vlm_model,
                    field_id_hint=field_id_hint,
                )
            )

        return RightChannelPageOutput(
            page_index=page_index,
            form_title=self._to_optional_string(payload.get("form_title")) if isinstance(payload, dict) else None,
            form_type=self._to_optional_string(payload.get("form_type")) if isinstance(payload, dict) else None,
            semantic_candidates=semantic_candidates,
        )

    def _parse_field_type(self, value: object) -> FieldType:
        """解析字段类型。"""

        normalized = str(value or "").strip().lower()
        for field_type in FieldType:
            if field_type.value == normalized:
                return field_type
        return FieldType.TEXT

    def _parse_region_hint(self, value: object) -> RegionHint | None:
        """解析区域提示对象。"""

        if not isinstance(value, dict):
            return None
        return RegionHint(
            anchor_text=self._to_optional_string(value.get("anchor_text")),
            direction=self._to_optional_string(value.get("direction")),
            priority_zone=self._to_optional_string(value.get("priority_zone")),
        )

    def _to_optional_string(self, value: object) -> str | None:
        """将任意值转为可选字符串。"""

        text = str(value or "").strip()
        return text or None

    def _to_optional_bool(self, value: object) -> bool | None:
        """将任意值转为可选布尔值。"""

        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return None

    def _to_confidence(self, value: object) -> float:
        """规范化置信度值。"""

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        return max(0.0, min(1.0, confidence))
