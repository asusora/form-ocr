"""应用配置定义。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用运行配置。"""

    app_name: str = Field(default="Form OCR M1", alias="FORM_OCR_APP_NAME")
    api_v1_prefix: str = Field(default="/api/v1", alias="FORM_OCR_API_V1_PREFIX")
    data_dir: str = Field(default="artifacts", alias="FORM_OCR_DATA_DIR")
    templates_path: str = Field(
        default="config/template_registry.json",
        alias="FORM_OCR_TEMPLATES_PATH",
    )
    key_alias_registry_path: str = Field(
        default="config/key_alias_registry.json",
        alias="FORM_OCR_KEY_ALIAS_REGISTRY_PATH",
    )
    pipeline_version: str = Field(default="v1", alias="FORM_OCR_PIPELINE_VERSION")
    render_dpi: int = Field(default=300, alias="FORM_OCR_RENDER_DPI")
    max_upload_mb: int = Field(default=25, alias="FORM_OCR_MAX_UPLOAD_MB")
    anchor_confidence_threshold: float = Field(
        default=0.65,
        alias="FORM_OCR_ANCHOR_CONFIDENCE_THRESHOLD",
    )
    template_hit_threshold: float = Field(
        default=0.85,
        alias="FORM_OCR_TEMPLATE_HIT_THRESHOLD",
    )
    template_partial_threshold: float = Field(
        default=0.50,
        alias="FORM_OCR_TEMPLATE_PARTIAL_THRESHOLD",
    )
    template_layout_weight: float = Field(
        default=0.35,
        alias="FORM_OCR_TEMPLATE_LAYOUT_WEIGHT",
    )
    template_anchor_weight: float = Field(
        default=0.45,
        alias="FORM_OCR_TEMPLATE_ANCHOR_WEIGHT",
    )
    template_page_count_weight: float = Field(
        default=0.20,
        alias="FORM_OCR_TEMPLATE_PAGE_COUNT_WEIGHT",
    )
    line_min_width_ratio: float = Field(
        default=0.08,
        alias="FORM_OCR_LINE_MIN_WIDTH_RATIO",
    )
    line_max_width_ratio: float = Field(
        default=0.90,
        alias="FORM_OCR_LINE_MAX_WIDTH_RATIO",
    )
    line_max_height_ratio: float = Field(
        default=0.03,
        alias="FORM_OCR_LINE_MAX_HEIGHT_RATIO",
    )
    checkbox_min_area: int = Field(
        default=100,
        alias="FORM_OCR_CHECKBOX_MIN_AREA",
    )
    checkbox_max_area: int = Field(
        default=5000,
        alias="FORM_OCR_CHECKBOX_MAX_AREA",
    )
    checkbox_fill_threshold: float = Field(
        default=0.30,
        alias="FORM_OCR_CHECKBOX_FILL_THRESHOLD",
    )
    text_filled_ink_threshold: float = Field(
        default=0.02,
        alias="FORM_OCR_TEXT_FILLED_INK_THRESHOLD",
    )
    ocr_engine: str = Field(default="rapidocr", alias="FORM_OCR_OCR_ENGINE")
    vlm_base_url: str = Field(
        default="http://127.0.0.1:8011/v1",
        alias="FORM_OCR_VLM_BASE_URL",
    )
    vlm_api_key: str = Field(
        default="sk-test",
        alias="FORM_OCR_VLM_API_KEY",
    )
    vlm_model: str = Field(
        default="Qwen2-VL-7B-Instruct",
        alias="FORM_OCR_VLM_MODEL",
    )
    vlm_timeout_seconds: float = Field(
        default=60.0,
        alias="FORM_OCR_VLM_TIMEOUT_SECONDS",
    )
    llm_base_url: str = Field(
        default="http://192.168.51.202:9098/v1",
        alias="FORM_OCR_LLM_BASE_URL",
    )
    llm_api_key: str = Field(
        default="",
        alias="FORM_OCR_LLM_API_KEY",
    )
    llm_model: str = Field(
        default="Qwopus3.5",
        alias="FORM_OCR_LLM_MODEL",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        alias="FORM_OCR_LLM_TIMEOUT_SECONDS",
    )
    default_vlm_model: str = Field(
        default="Qwen2-VL-7B-Instruct",
        alias="FORM_OCR_DEFAULT_VLM_MODEL",
    )
    cors_origins: str = Field(
        default="http://127.0.0.1:5173,http://localhost:5173",
        alias="FORM_OCR_CORS_ORIGINS",
    )
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def project_root(self) -> Path:
        """返回项目根目录。"""

        return Path(__file__).resolve().parents[2]

    @property
    def artifacts_root(self) -> Path:
        """返回工件根目录。"""

        return (self.project_root / self.data_dir).resolve()

    @property
    def templates_registry_path(self) -> Path:
        """返回模板注册表路径。"""

        return (self.project_root / self.templates_path).resolve()

    @property
    def key_alias_registry_file_path(self) -> Path:
        """返回字段别名注册表路径。"""

        return (self.project_root / self.key_alias_registry_path).resolve()

    @property
    def cors_origin_list(self) -> list[str]:
        """返回解析后的跨域白名单。"""

        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回缓存后的配置对象。"""

    return Settings()
