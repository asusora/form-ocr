"""服务容器定义。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.repositories.task_repository import TaskRepository
from app.repositories.template_repository import TemplateRepository
from app.services.export_service import ExportService
from app.services.left_channel_service import LeftChannelService
from app.services.ocr_service import BaseOcrEngine, build_ocr_engine
from app.services.pipeline_service import PipelineService
from app.services.preprocess_service import PreprocessService
from app.services.router_service import RouterService
from app.services.semantic_placeholder import SemanticPlaceholderService


@dataclass
class ServiceContainer:
    """应用服务容器。"""

    settings: Settings
    task_repository: TaskRepository
    template_repository: TemplateRepository
    ocr_engine: BaseOcrEngine
    semantic_service: SemanticPlaceholderService
    preprocess_service: PreprocessService
    router_service: RouterService
    left_channel_service: LeftChannelService
    export_service: ExportService
    pipeline_service: PipelineService


def build_container(settings: Settings | None = None) -> ServiceContainer:
    """构建服务容器。"""

    resolved_settings = settings or get_settings()
    task_repository = TaskRepository(resolved_settings)
    template_repository = TemplateRepository(resolved_settings)
    ocr_engine = build_ocr_engine(resolved_settings)
    semantic_service = SemanticPlaceholderService(resolved_settings)
    preprocess_service = PreprocessService(resolved_settings, task_repository, ocr_engine)
    router_service = RouterService(resolved_settings, template_repository)
    left_channel_service = LeftChannelService(resolved_settings, task_repository, ocr_engine)
    export_service = ExportService(resolved_settings, task_repository)
    pipeline_service = PipelineService(
        repository=task_repository,
        preprocess_service=preprocess_service,
        router_service=router_service,
        left_channel_service=left_channel_service,
        export_service=export_service,
        max_upload_mb=resolved_settings.max_upload_mb,
    )
    return ServiceContainer(
        settings=resolved_settings,
        task_repository=task_repository,
        template_repository=template_repository,
        ocr_engine=ocr_engine,
        semantic_service=semantic_service,
        preprocess_service=preprocess_service,
        router_service=router_service,
        left_channel_service=left_channel_service,
        export_service=export_service,
        pipeline_service=pipeline_service,
    )
