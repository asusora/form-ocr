"""管线编排服务。"""

from __future__ import annotations

import time

from app.core.errors import FormOcrException
from app.models.common import TaskContext, TaskStatus
from app.models.stages import ExportOutput
from app.repositories.task_repository import TaskRepository
from app.services.export_service import ExportService
from app.services.fusion_service import FusionService
from app.services.key_mapping_service import KeyMappingService
from app.services.left_channel_service import LeftChannelService
from app.services.preprocess_service import PreprocessService
from app.services.right_channel_service import RightChannelService
from app.services.router_service import RouterService


class PipelineService:
    """串联 M1 与 M2 管线。"""

    ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        *,
        repository: TaskRepository,
        preprocess_service: PreprocessService,
        router_service: RouterService,
        left_channel_service: LeftChannelService,
        right_channel_service: RightChannelService,
        fusion_service: FusionService,
        key_mapping_service: KeyMappingService,
        export_service: ExportService,
        max_upload_mb: int,
    ) -> None:
        """注入依赖。"""

        self.repository = repository
        self.preprocess_service = preprocess_service
        self.router_service = router_service
        self.left_channel_service = left_channel_service
        self.right_channel_service = right_channel_service
        self.fusion_service = fusion_service
        self.key_mapping_service = key_mapping_service
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
        task_context, artifact_version = self._prepare_run(task_id)

        stage0_output = self.preprocess_service.run(task_context)
        self.repository.save_stage_output(task_id, "stage0_output", stage0_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.PREPROCESSED, page_count=len(stage0_output.pages))

        route_decision = self.router_service.decide(stage0_output)
        self.repository.save_stage_output(task_id, "route_decision", route_decision, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.ROUTED)

        left_output = self.left_channel_service.run(stage0_output, route_decision)
        self.repository.save_stage_output(task_id, "left_channel_output", left_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.DETECTED)

        keyed_output = self.key_mapping_service.build_from_left_output(left_output, route_decision)
        export_output = self.export_service.run(
            self.repository.get_task_context(task_id),
            stage0_output,
            route_decision,
            keyed_output,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
        )
        self.repository.save_stage_output(task_id, "export_output", export_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.EXPORTED)
        return export_output

    def run_m2(self, task_id: str) -> ExportOutput:
        """执行 M2 语义增强闭环。"""

        start_time = time.perf_counter()
        task_context, artifact_version = self._prepare_run(task_id)

        stage0_output = self.preprocess_service.run(task_context)
        self.repository.save_stage_output(task_id, "stage0_output", stage0_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.PREPROCESSED, page_count=len(stage0_output.pages))

        route_decision = self.router_service.decide(stage0_output)
        self.repository.save_stage_output(task_id, "route_decision", route_decision, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.ROUTED)

        left_output = self.left_channel_service.run(stage0_output, route_decision)
        self.repository.save_stage_output(task_id, "left_channel_output", left_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.DETECTED)

        right_output = self.right_channel_service.run(stage0_output, route_decision, left_output)
        self.repository.save_stage_output(task_id, "right_channel_output", right_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.SEMANTICS_READY)

        fusion_output = self.fusion_service.run(left_output, right_output, route_decision)
        self.repository.save_stage_output(task_id, "fusion_output", fusion_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.MERGED)

        keyed_output = self.key_mapping_service.run(fusion_output, route_decision)
        self.repository.save_stage_output(task_id, "key_mapping_output", keyed_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.KEYED)

        export_output = self.export_service.run(
            self.repository.get_task_context(task_id),
            stage0_output,
            route_decision,
            keyed_output,
            processing_time_ms=int((time.perf_counter() - start_time) * 1000),
        )
        self.repository.save_stage_output(task_id, "export_output", export_output, artifact_version=artifact_version)
        self.repository.update_task_status(task_id, TaskStatus.EXPORTED)
        return export_output

    def _prepare_run(self, task_id: str) -> tuple[TaskContext, int]:
        """准备本次执行所需的任务上下文与工件版本。"""

        task_context = self.repository.prepare_artifact_version(task_id)
        return task_context, task_context.artifact_version
