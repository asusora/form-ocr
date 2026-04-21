"""公共数据模型定义。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    """基础模型配置。"""

    model_config = ConfigDict(populate_by_name=True, use_enum_values=True)


class FieldType(str, Enum):
    """字段类型枚举。"""

    TEXT = "text"
    DATE = "date"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"
    HANDWRITING = "handwriting"


class TaskStatus(str, Enum):
    """任务状态枚举。"""

    UPLOADED = "uploaded"
    RENDERED = "rendered"
    PREPROCESSED = "preprocessed"
    ROUTED = "routed"
    DETECTED = "detected"
    EXPORTED = "exported"


class RouteType(str, Enum):
    """路由类型枚举。"""

    TEMPLATE_HIT = "template_hit"
    TEMPLATE_PARTIAL = "template_partial"
    TEMPLATE_UNKNOWN = "template_unknown"


class KeySource(str, Enum):
    """字段键来源枚举。"""

    TEMPLATE_CANONICAL = "template_canonical"
    VLM_RAW = "vlm_raw"
    NEW_FIELD = "new_field"
    USER_DEFINED = "user_defined"


class ValueSource(str, Enum):
    """字段值来源枚举。"""

    OCR = "ocr"
    VLM = "vlm"
    VLM_CORRECTED = "vlm_corrected"
    CROP_ONLY = "crop_only"
    USER_INPUT = "user_input"


class ConfidenceLevel(str, Enum):
    """置信度等级枚举。"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BBox(SchemaModel):
    """归一化矩形框。"""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(ge=0.0, le=1.0)
    h: float = Field(ge=0.0, le=1.0)


class TaskContext(SchemaModel):
    """任务上下文。"""

    task_id: str
    tenant_id: str = "tenant_default"
    source_file_name: str
    source_file_type: str
    source_file_url: str
    page_count: int = 0
    task_status: TaskStatus = TaskStatus.UPLOADED
    pipeline_version: str
    artifact_version: int = 1
    created_at: str
    updated_at: str


class PageAsset(SchemaModel):
    """页面工件信息。"""

    page_id: str
    task_id: str
    page_index: int
    width: int
    height: int
    dpi: int
    origin_image_url: str
    preprocessed_image_url: str
    rotation_degree: float = 0.0
    deskew_applied: bool = False


class AnchorText(SchemaModel):
    """锚点文本对象。"""

    page_id: str
    anchor_id: str
    text: str
    bbox: BBox
    confidence: float
    language: str


class MatchedTemplate(SchemaModel):
    """模板命中信息。"""

    template_id: str
    template_revision: int
    confidence: float


class KnownFieldRegion(SchemaModel):
    """模板中的已知字段区域。"""

    canonical_key_id: Optional[str] = None
    field_type: FieldType
    bbox_relative: BBox
    key: Optional[str] = None
    key_en: Optional[str] = None


class PageRoute(SchemaModel):
    """单页路由结果。"""

    page_index: int
    route_type: RouteType
    template_id: Optional[str] = None
    template_revision: Optional[int] = None
    affine_ready: bool = False
    known_field_regions: list[KnownFieldRegion] = Field(default_factory=list)
    unknown_region_strategy: Optional[str] = None


class HealthResponse(SchemaModel):
    """健康检查响应。"""

    status: str
    app_name: str
    pipeline_version: str


class LlmConfigStatus(SchemaModel):
    """大模型配置状态。"""

    default_vlm_model: str
    openai_configured: bool
    anthropic_configured: bool
