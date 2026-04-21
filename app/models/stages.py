"""各阶段输入输出模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from app.models.common import (
    ConfidenceLevel,
    FieldType,
    KeySource,
    MatchedTemplate,
    PageAsset,
    PageRoute,
    RouteType,
    SchemaModel,
    TaskStatus,
    ValueSource,
    AnchorText,
    BBox,
)


class Stage0Output(SchemaModel):
    """阶段 0 输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.PREPROCESSED
    pages: list[PageAsset] = Field(default_factory=list)
    anchors: list[AnchorText] = Field(default_factory=list)


class RouteDecision(SchemaModel):
    """路由阶段输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.ROUTED
    route_type: RouteType
    matched_template: Optional[MatchedTemplate] = None
    page_routes: list[PageRoute] = Field(default_factory=list)
    reason: str


class FieldCandidate(SchemaModel):
    """左通道字段候选。"""

    field_id: str
    task_id: str
    page_index: int
    field_type: FieldType
    bbox: BBox
    crop_bbox: BBox
    detector_source: str
    crop_image_url: str
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    is_filled: bool = False
    ink_ratio: float = 0.0
    line_anchor_text: Optional[str] = None
    is_checked: Optional[bool] = None
    canonical_key_id: Optional[str] = None
    key_hint: Optional[str] = None
    key_hint_en: Optional[str] = None


class LeftChannelPageOutput(SchemaModel):
    """单页左通道输出。"""

    page_index: int
    field_candidates: list[FieldCandidate] = Field(default_factory=list)


class LeftChannelOutput(SchemaModel):
    """左通道总输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.DETECTED
    pages: list[LeftChannelPageOutput] = Field(default_factory=list)


class ExportPage(SchemaModel):
    """导出页面对象。"""

    page_index: int
    page_image_url: str
    page_preprocessed_image_url: str


class ExportedField(SchemaModel):
    """导出字段对象。"""

    field_id: str
    page_index: int
    canonical_key_id: Optional[str] = None
    key: str
    key_en: Optional[str] = None
    key_source: KeySource
    key_raw: str
    value: Any = None
    value_source: ValueSource
    confidence: float
    confidence_level: ConfidenceLevel
    bbox: BBox
    origin_bbox: Optional[BBox] = None
    crop_bbox: Optional[BBox] = None
    crop_image_url: str
    field_type: FieldType
    is_filled: bool
    dirty: bool = False
    dirty_reason: Optional[str] = None
    review_required: bool = False
    user_modified: bool = False
    user_confirmed: bool = False


class KvPairEntry(SchemaModel):
    """业务系统向 KV 视图。"""

    display_name: str
    value: Any = None
    field_type: FieldType
    page_index: int
    field_id: str


class ExportMetadata(SchemaModel):
    """导出元数据。"""

    processing_time_ms: int
    ocr_engine: str
    vlm_model: Optional[str] = None
    template_matched: bool
    artifact_version: int
    keys_from_template: int
    keys_from_vlm_raw: int
    corrections_made: int = 0


class ExportOutput(SchemaModel):
    """阶段 4 导出对象。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.EXPORTED
    template_id: Optional[str] = None
    template_revision: Optional[int] = None
    template_match_confidence: Optional[float] = None
    pages: list[ExportPage] = Field(default_factory=list)
    fields: list[ExportedField] = Field(default_factory=list)
    kv_pairs: dict[str, KvPairEntry] = Field(default_factory=dict)
    metadata: ExportMetadata


class TemplateAnchor(SchemaModel):
    """模板锚点。"""

    page_index: int
    text: str
    bbox: BBox


class TemplateField(SchemaModel):
    """模板字段定义。"""

    page_index: int
    canonical_key_id: str
    field_type: FieldType
    bbox_relative: BBox
    key: Optional[str] = None
    key_en: Optional[str] = None


class TemplateDefinition(SchemaModel):
    """模板定义。"""

    template_id: str
    template_revision: int
    form_title: Optional[str] = None
    page_count: int
    anchors: list[TemplateAnchor] = Field(default_factory=list)
    fields: list[TemplateField] = Field(default_factory=list)


class TemplateRegistry(SchemaModel):
    """模板注册表。"""

    templates: list[TemplateDefinition] = Field(default_factory=list)
