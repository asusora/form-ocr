"""本地 OpenAI 兼容模型客户端。"""

from __future__ import annotations

import json
import re
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.parse import quote

import httpx

from app.core.errors import FormOcrException


JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


class OpenAiCompatibleClient:
    """封装本地 OpenAI 兼容 `chat/completions` 接口。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        client_factory: Callable[..., httpx.Client] | None = None,
    ) -> None:
        """记录客户端配置。"""

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout_seconds = timeout_seconds
        self.client_factory = client_factory or httpx.Client

    @property
    def configured(self) -> bool:
        """返回当前客户端是否可用。"""

        return bool(self.base_url)

    def chat_completions(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """调用 OpenAI 兼容聊天补全接口。"""

        if not self.configured:
            raise FormOcrException(
                code="MODEL_NOT_CONFIGURED",
                message="本地模型服务未配置。",
                status_code=500,
            )
        if not model.strip():
            raise FormOcrException(
                code="MODEL_NOT_CONFIGURED",
                message="本地模型名称为空。",
                status_code=500,
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            with self.client_factory(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException as exc:
            raise FormOcrException(
                code="MODEL_TIMEOUT",
                message="本地模型请求超时。",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise FormOcrException(
                code="MODEL_REQUEST_FAILED",
                message=f"本地模型请求失败: {exc}",
                status_code=502,
            ) from exc
        except ValueError as exc:
            raise FormOcrException(
                code="MODEL_RESPONSE_INVALID",
                message="本地模型返回了非法 JSON。",
                status_code=502,
            ) from exc

    @staticmethod
    def extract_message_text(payload: dict[str, Any]) -> str:
        """从聊天补全响应中提取文本内容。"""

        choices = payload.get("choices") or []
        if not choices:
            return ""

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
            return "\n".join(part for part in text_parts if part).strip()
        return ""

    @staticmethod
    def extract_json_payload(text: str) -> Any:
        """从模型文本中提取 JSON 对象。"""

        normalized = (text or "").strip()
        if not normalized:
            raise FormOcrException(
                code="MODEL_RESPONSE_EMPTY",
                message="本地模型返回为空。",
                status_code=502,
            )

        fenced_match = JSON_BLOCK_RE.search(normalized)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        try:
            return json.loads(normalized)
        except ValueError:
            pass

        start_positions = [index for index in (normalized.find("{"), normalized.find("[")) if index >= 0]
        if not start_positions:
            raise FormOcrException(
                code="MODEL_RESPONSE_INVALID",
                message="本地模型未返回 JSON 内容。",
                status_code=502,
            )
        start_index = min(start_positions)
        closing_char = "}" if normalized[start_index] == "{" else "]"
        end_index = normalized.rfind(closing_char)
        if end_index <= start_index:
            raise FormOcrException(
                code="MODEL_RESPONSE_INVALID",
                message="本地模型返回的 JSON 不完整。",
                status_code=502,
            )
        return json.loads(normalized[start_index:end_index + 1])


def build_local_file_url(path: str | Path) -> str:
    """将本地文件路径转换为 `file:///` URL。"""

    if isinstance(path, Path):
        try:
            return path.resolve().as_uri()
        except ValueError:
            normalized_path = str(path)
    else:
        normalized_path = str(path)

    if re.match(r"^[A-Za-z]:[\\/]", normalized_path):
        windows_path = PureWindowsPath(normalized_path)
        drive = windows_path.drive.rstrip(":")
        parts = [quote(part) for part in windows_path.parts[1:]]
        return f"file:///{drive}:/" + "/".join(parts)

    return Path(normalized_path).resolve().as_uri()
