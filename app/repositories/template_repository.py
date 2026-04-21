"""模板注册表仓储实现。"""

from __future__ import annotations

import json

from app.core.config import Settings
from app.models.stages import TemplateRegistry


class TemplateRepository:
    """负责读取模板注册表。"""

    def __init__(self, settings: Settings) -> None:
        """确保模板配置文件存在。"""

        self.settings = settings
        self.settings.templates_registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.settings.templates_registry_path.exists():
            self.settings.templates_registry_path.write_text(
                json.dumps({"templates": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_registry(self) -> TemplateRegistry:
        """读取模板注册表。"""

        content = self.settings.templates_registry_path.read_text(encoding="utf-8")
        return TemplateRegistry.model_validate_json(content)
