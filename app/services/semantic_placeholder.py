"""大模型配置占位服务。"""

from app.core.config import Settings
from app.models.common import LlmConfigStatus


class SemanticPlaceholderService:
    """仅暴露大模型配置状态，不在 M1 中实际调用。"""

    def __init__(self, settings: Settings) -> None:
        """记录当前配置。"""

        self.settings = settings

    def get_status(self) -> LlmConfigStatus:
        """返回大模型配置是否已就绪。"""

        return LlmConfigStatus(
            default_vlm_model=self.settings.default_vlm_model,
            openai_configured=bool(self.settings.openai_api_key.strip()),
            anthropic_configured=bool(self.settings.anthropic_api_key.strip()),
        )
