"""各阶段输入输出模型。"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from app.models.common import (
    AnchorText,
    BBox,
    ConfidenceLevel,
    FieldType,
    KeySource,
    MatchStatus,
    MatchedTemplate,
    PageAsset,
    PageRoute,
    RouteType,
    SchemaModel,
    TaskStatus,
    ValueSource,
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


class RegionHint(SchemaModel):
    """右通道输出的区域提示。"""

    anchor_text: Optional[str] = None
    direction: Optional[str] = None
    priority_zone: Optional[str] = None


class SemanticFieldCandidate(SchemaModel):
    """右通道语义候选。"""

    semantic_candidate_id: str
    task_id: str
    page_index: int
    semantic_label: str
    field_type: FieldType
    value_read: Optional[str] = None
    is_signed: Optional[bool] = None
    near_anchor_text: Optional[str] = None
    spatial_description: Optional[str] = None
    region_hint: Optional[RegionHint] = None
    semantic_confidence: float = 0.0
    model_name: str
    field_id_hint: Optional[str] = None


class RightChannelPageOutput(SchemaModel):
    """单页右通道输出。"""

    page_index: int
    form_title: Optional[str] = None
    form_type: Optional[str] = None
    semantic_candidates: list[SemanticFieldCandidate] = Field(default_factory=list)


class RightChannelOutput(SchemaModel):
    """右通道总输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.SEMANTICS_READY
    form_title: Optional[str] = None
    form_type: Optional[str] = None
    pages: list[RightChannelPageOutput] = Field(default_factory=list)


class FieldCorrection(SchemaModel):
    """字段纠错说明。"""

    ocr_original: Optional[str] = None
    vlm_original: Optional[str] = None
    final_value: Any = None
    reason: Optional[str] = None


class MergedField(SchemaModel):
    """融合后的统一字段对象。"""

    field_id: str
    task_id: str
    page_index: int
    field_type: FieldType
    bbox: BBox
    crop_bbox: BBox
    crop_image_url: str
    canonical_key_id: Optional[str] = None
    key_hint: Optional[str] = None
    key_hint_en: Optional[str] = None
    line_anchor_text: Optional[str] = None
    semantic_label: Optional[str] = None
    key_raw: str
    value: Any = None
    key_source: KeySource
    value_source: ValueSource
    confidence: float
    confidence_level: ConfidenceLevel
    match_status: MatchStatus
    review_required: bool = False
    correction: Optional[FieldCorrection] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    is_filled: bool = False
    ink_ratio: float = 0.0
    is_checked: Optional[bool] = None


class UnresolvedCandidate(SchemaModel):
    """未能自动闭环的右通道候选。"""

    semantic_candidate_id: str
    page_index: int
    field_type: FieldType
    semantic_label: str
    reason: str
    resolution: str


class FusionOutput(SchemaModel):
    """阶段 2 融合输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.MERGED
    form_title: Optional[str] = None
    form_type: Optional[str] = None
    merged_fields: list[MergedField] = Field(default_factory=list)
    unresolved_candidates: list[UnresolvedCandidate] = Field(default_factory=list)


class PendingKey(SchemaModel):
    """待确认字段映射。"""

    field_id: str
    page_index: int
    field_type: FieldType
    key_raw: str
    suggested_canonical_key_name: Optional[str] = None
    reason: str


class KeyedField(SchemaModel):
    """阶段 3 产出的已映射字段。"""

    field_id: str
    task_id: str
    page_index: int
    field_type: FieldType
    canonical_key_id: Optional[str] = None
    canonical_key: Optional[str] = None
    canonical_key_en: Optional[str] = None
    key: str
    key_en: Optional[str] = None
    key_raw: str
    key_source: KeySource
    value: Any = None
    value_source: ValueSource
    confidence: float
    confidence_level: ConfidenceLevel
    bbox: BBox
    crop_bbox: BBox
    crop_image_url: str
    is_filled: bool
    review_required: bool = False
    mapping_reason: Optional[str] = None
    semantic_label: Optional[str] = None
    match_status: MatchStatus
    correction: Optional[FieldCorrection] = None
    origin_bbox: Optional[BBox] = None
    dirty: bool = False
    dirty_reason: Optional[str] = None
    user_modified: bool = False
    user_confirmed: bool = False


class KeyMappingOutput(SchemaModel):
    """阶段 3 输出。"""

    task_id: str
    task_status: TaskStatus = TaskStatus.KEYED
    form_title: Optional[str] = None
    form_type: Optional[str] = None
    keyed_fields: list[KeyedField] = Field(default_factory=list)
    pending_keys: list[PendingKey] = Field(default_factory=list)


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
    semantic_label: Optional[str] = None
    match_status: Optional[MatchStatus] = None
    mapping_reason: Optional[str] = None
    correction: Optional[FieldCorrection] = None


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
    form_title: Optional[str] = None
    form_type: Optional[str] = None
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


class KeyAliasDefinition(SchemaModel):
    """字段别名定义。"""

    canonical_key_id: str
    canonical_key: str
    canonical_key_en: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)


class KeyAliasRegistry(SchemaModel):
    """字段别名注册表。"""

    keys: list[KeyAliasDefinition] = Field(default_factory=list)
