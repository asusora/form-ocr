"""本地模型配置状态服务。"""

from app.core.config import Settings
from app.models.common import LlmConfigStatus


class SemanticPlaceholderService:
    """暴露本地 LLM/VLM 配置状态。"""

    def __init__(self, settings: Settings) -> None:
        """记录当前配置。"""

        self.settings = settings

    def get_status(self) -> LlmConfigStatus:
        """返回本地模型配置是否已就绪。"""

        return LlmConfigStatus(
            vlm_configured=bool(self.settings.vlm_base_url.strip() and self.settings.vlm_model.strip()),
            vlm_base_url=self.settings.vlm_base_url,
            vlm_model=self.settings.vlm_model,
            llm_configured=bool(self.settings.llm_base_url.strip() and self.settings.llm_model.strip()),
            llm_base_url=self.settings.llm_base_url,
            llm_model=self.settings.llm_model,
        )
