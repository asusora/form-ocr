"""任务与工件仓储实现。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.errors import FormOcrException
from app.core.time_utils import now_iso
from app.models.common import TaskContext, TaskStatus
from app.utils.id_utils import generate_task_id


class TaskRepository:
    """负责本地任务与工件持久化。"""

    TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
    RELATIVE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(self, settings: Settings) -> None:
        """初始化仓储目录。"""

        self.settings = settings
        self.tasks_root = self.settings.artifacts_root / "tasks"
        self.tasks_root.mkdir(parents=True, exist_ok=True)

    def create_task(
        self,
        source_file_name: str,
        source_file_type: str,
        file_bytes: bytes,
    ) -> TaskContext:
        """创建任务并保存原始文件。"""

        task_id = generate_task_id()
        task_dir = self.get_task_dir(task_id)
        for relative in ("source", "pages", "fields", "results", "debug"):
            (task_dir / relative).mkdir(parents=True, exist_ok=True)

        suffix = Path(source_file_name).suffix.lower() or ".bin"
        source_path = task_dir / "source" / f"original{suffix}"
        source_path.write_bytes(file_bytes)

        timestamp = now_iso()
        context = TaskContext(
            task_id=task_id,
            source_file_name=source_file_name,
            source_file_type=source_file_type,
            source_file_url=self.path_to_url(source_path),
            page_count=0,
            task_status=TaskStatus.UPLOADED,
            pipeline_version=self.settings.pipeline_version,
            artifact_version=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.save_task_context(context)
        return context

    def get_task_dir(self, task_id: str) -> Path:
        """返回任务目录。"""

        self._validate_task_id(task_id)
        return self.tasks_root / task_id

    def path_to_url(self, path: Path) -> str:
        """将工件路径转换为静态访问 URL。"""

        relative = path.resolve().relative_to(self.settings.artifacts_root.resolve())
        return f"/artifacts/{relative.as_posix()}"

    def url_to_path(self, url: str) -> Path:
        """将工件 URL 转换为本地路径。"""

        normalized = url.lstrip("/")
        if not normalized.startswith("artifacts/"):
            raise FormOcrException(
                code="EXPORT_ARTIFACT_FAILED",
                message=f"非法工件 URL: {url}",
                status_code=400,
            )
        relative = normalized.replace("artifacts/", "", 1)
        resolved = (self.settings.artifacts_root / relative).resolve()
        self._ensure_within_root(resolved, self.settings.artifacts_root.resolve(), "EXPORT_ARTIFACT_FAILED")
        return resolved

    def build_artifact_path(self, task_id: str, *relative_parts: str) -> Path:
        """构建任务内工件路径。"""

        task_dir = self.get_task_dir(task_id).resolve()
        for part in relative_parts:
            self._validate_relative_part(part)
        path = task_dir.joinpath(*relative_parts).resolve()
        self._ensure_within_root(path, task_dir, "EXPORT_ARTIFACT_FAILED")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def build_artifact_url(self, task_id: str, *relative_parts: str) -> str:
        """构建任务内工件 URL。"""

        return self.path_to_url(self.build_artifact_path(task_id, *relative_parts))

    def save_task_context(self, context: TaskContext) -> TaskContext:
        """保存任务上下文。"""

        context.updated_at = now_iso()
        payload = context.model_dump(mode="json")
        path = self.build_artifact_path(context.task_id, "task.json")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return context

    def get_task_context(self, task_id: str) -> TaskContext:
        """读取任务上下文。"""

        path = self.build_artifact_path(task_id, "task.json")
        if not path.exists():
            raise FormOcrException(
                code="TASK_NOT_FOUND",
                message=f"任务不存在: {task_id}",
                status_code=404,
            )
        return TaskContext.model_validate_json(path.read_text(encoding="utf-8"))

    def update_task_status(
        self,
        task_id: str,
        task_status: TaskStatus,
        *,
        page_count: int | None = None,
        artifact_version: int | None = None,
    ) -> TaskContext:
        """更新任务状态。"""

        context = self.get_task_context(task_id)
        context.task_status = task_status
        if page_count is not None:
            context.page_count = page_count
        if artifact_version is not None:
            context.artifact_version = artifact_version
        return self.save_task_context(context)

    def prepare_artifact_version(self, task_id: str) -> TaskContext:
        """为新的执行轮次准备工件版本。"""

        context = self.get_task_context(task_id)
        if context.task_status == TaskStatus.UPLOADED and context.artifact_version == 1:
            return context

        next_version = max(1, context.artifact_version + 1)
        return self.update_task_status(
            task_id,
            context.task_status,
            page_count=context.page_count,
            artifact_version=next_version,
        )

    def save_stage_output(
        self,
        task_id: str,
        stage_name: str,
        payload: Any,
        *,
        artifact_version: int | None = None,
    ) -> Path:
        """保存阶段输出 JSON。"""

        version = self._resolve_artifact_version(task_id, artifact_version)
        serializable = self._to_serializable(payload)
        versioned_path = self.build_artifact_path(task_id, "debug", f"v{version}", f"{stage_name}.json")
        legacy_path = self.build_artifact_path(task_id, "debug", f"{stage_name}.json")
        self._write_json(versioned_path, serializable)
        self._write_json(legacy_path, serializable)
        return versioned_path

    def load_stage_output(
        self,
        task_id: str,
        stage_name: str,
        *,
        artifact_version: int | None = None,
    ) -> dict[str, Any]:
        """读取阶段输出 JSON。"""

        version = self._resolve_artifact_version(task_id, artifact_version)
        candidates = [
            self.build_artifact_path(task_id, "debug", f"v{version}", f"{stage_name}.json"),
            self.build_artifact_path(task_id, "debug", f"{stage_name}.json"),
        ]
        for stage_path in candidates:
            if stage_path.exists():
                return json.loads(stage_path.read_text(encoding="utf-8"))
        raise FormOcrException(
            code="ARTIFACT_NOT_FOUND",
            message=f"阶段结果不存在: {stage_name}",
            status_code=404,
        )

    def save_result_output(
        self,
        task_id: str,
        result_name: str,
        payload: Any,
        *,
        artifact_version: int | None = None,
    ) -> Path:
        """保存结果输出 JSON。"""

        version = self._resolve_artifact_version(task_id, artifact_version)
        serializable = self._to_serializable(payload)
        versioned_path = self.build_artifact_path(task_id, "results", f"v{version}", f"{result_name}.json")
        legacy_path = self.build_artifact_path(task_id, "results", f"{result_name}.json")
        self._write_json(versioned_path, serializable)
        self._write_json(legacy_path, serializable)
        return versioned_path

    def load_result_output(
        self,
        task_id: str,
        result_name: str,
        *,
        artifact_version: int | None = None,
    ) -> dict[str, Any]:
        """读取结果输出 JSON。"""

        version = self._resolve_artifact_version(task_id, artifact_version)
        candidates = [
            self.build_artifact_path(task_id, "results", f"v{version}", f"{result_name}.json"),
            self.build_artifact_path(task_id, "results", f"{result_name}.json"),
        ]
        for result_path in candidates:
            if result_path.exists():
                return json.loads(result_path.read_text(encoding="utf-8"))
        raise FormOcrException(
            code="ARTIFACT_NOT_FOUND",
            message=f"结果文件不存在: {result_name}",
            status_code=404,
        )

    def _validate_task_id(self, task_id: str) -> None:
        """校验任务标识，避免目录穿越。"""

        if not self.TASK_ID_PATTERN.fullmatch(task_id):
            raise FormOcrException(
                code="TASK_ID_INVALID",
                message=f"非法任务标识: {task_id}",
                status_code=400,
            )

    def _validate_relative_part(self, value: str) -> None:
        """校验相对路径片段。"""

        if not value or not self.RELATIVE_NAME_PATTERN.fullmatch(value):
            raise FormOcrException(
                code="EXPORT_ARTIFACT_FAILED",
                message=f"非法工件路径片段: {value}",
                status_code=400,
            )

    def _ensure_within_root(self, candidate: Path, root: Path, error_code: str) -> None:
        """确保解析后的路径仍然位于指定根目录中。"""

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise FormOcrException(
                code=error_code,
                message=f"工件路径越界: {candidate}",
                status_code=400,
            ) from exc

    def _resolve_artifact_version(self, task_id: str, artifact_version: int | None) -> int:
        """解析要读写的工件版本号。"""

        if artifact_version is not None:
            return max(1, artifact_version)
        return max(1, self.get_task_context(task_id).artifact_version)

    def _to_serializable(self, payload: Any) -> Any:
        """将对象转换为可序列化数据。"""

        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="json")
        return payload

    def _write_json(self, path: Path, payload: Any) -> None:
        """写入 JSON 文件。"""

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
