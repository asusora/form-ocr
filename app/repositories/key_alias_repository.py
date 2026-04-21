"""字段别名注册表仓储实现。"""

from __future__ import annotations

import json

from app.core.config import Settings
from app.models.stages import KeyAliasRegistry


class KeyAliasRepository:
    """负责读取字段别名注册表。"""

    def __init__(self, settings: Settings) -> None:
        """确保字段别名配置文件存在。"""

        self.settings = settings
        self.settings.key_alias_registry_file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.settings.key_alias_registry_file_path.exists():
            self.settings.key_alias_registry_file_path.write_text(
                json.dumps({"keys": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load_registry(self) -> KeyAliasRegistry:
        """读取字段别名注册表。"""

        content = self.settings.key_alias_registry_file_path.read_text(encoding="utf-8")
        return KeyAliasRegistry.model_validate_json(content)
