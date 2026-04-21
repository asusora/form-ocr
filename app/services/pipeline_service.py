"""M1 管线编排服务。"""

from __future__ import annotations

import time

from app.core.errors import FormOcrException
from app.models.common import TaskContext, TaskStatus
from app.models.stages import ExportOutput
from app.repositories.task_repository import TaskRepository
from app.services.export_service import ExportService
from app.services.left_channel_service import LeftChannelService
from app.services.preprocess_service import PreprocessService
from app.services.router_service import RouterService


class PipelineService:
    """串联 M1 各阶段。"""

    ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        repository: TaskRepository,
        preprocess_service: PreprocessService,
        router_service: RouterService,
        left_channel_service: LeftChannelService,
        export_service: ExportService,
        max_upload_mb: int,
    ) -> None:
        """注入依赖。"""

        self.repository = repository
        self.preprocess_service = preprocess_service
        self.router_service = router_service
        self.left_channel_service = left_channel_service
        self.export_service = export_service
        self.max_upload_bytes = max_upload_mb * 1024 * 1024

    def create_task(self, file_name: str, content_type: str, file_bytes: bytes) -> TaskContext:
        """创建任务。"""

        suffix = file_name.lower().rsplit(".", maxsplit=1)
        extension = f".{suffix[-1]}" if len(suffix) > 1 else ""
        if extension not in self.ALLOWED_SUFFIXES:
            raise FormOcrException(
                code="INGESTION_FILE_UNSUPPORTED",
                message=f"不支持的文件格式: {extension or 'unknown'}",
                status_code=400,
            )
        if len(file_bytes) > self.max_upload_bytes:
            raise FormOcrException(
                code="INGESTION_FILE_TOO_LARGE",
                message=f"文件大小超过限制 {self.max_upload_bytes // (1024 * 1024)} MB",
                status_code=400,
            )
        return self.repository.create_task(file_name, content_type or "application/octet-stream", file_bytes)

    def run_m1(self, task_id: str) -> ExportOutput:
        """执行 M1 基础识别闭环。"""

        start_time = time.perf_counter()
        task_context = self.repository.get_task_context(task_id)

        stage0_output = self.preprocess_service.run(task_context)
        self.repository.save_stage_output(task_id, "stage0_output", stage0_output)
        self.repository.update_task_status(task_id, TaskStatus.PREPROCESSED, page_count=len(stage0_output.pages))

        route_decision = self.router_service.decide(stage0_output)
        self.repository.save_stage_output(task_id, "route_decision", route_decision)
        self.repository.update_task_status(task_id, TaskStatus.ROUTED)

        left_output = self.left_channel_service.run(stage0_output, route_decision)
        self.repository.save_stage_output(task_id, "left_channel_output", left_output)
        self.repository.update_task_status(task_id, TaskStatus.DETECTED)

        refreshed_context = self.repository.get_task_context(task_id)
        export_output = self.export_service.run(
            refreshed_context,
            stage0_output,
            route_decision,
            left_output,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
        )
        self.repository.save_stage_output(task_id, "export_output", export_output)
        self.repository.update_task_status(task_id, TaskStatus.EXPORTED)
        return export_output
