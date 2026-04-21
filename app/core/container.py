"""服务容器定义。"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.repositories.key_alias_repository import KeyAliasRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.template_repository import TemplateRepository
from app.services.export_service import ExportService
from app.services.fusion_service import FusionService
from app.services.key_mapping_service import KeyMappingService
from app.services.left_channel_service import LeftChannelService
from app.services.local_model_client import OpenAiCompatibleClient
from app.services.ocr_service import BaseOcrEngine, build_ocr_engine
from app.services.pipeline_service import PipelineService
from app.services.preprocess_service import PreprocessService
from app.services.right_channel_service import RightChannelService
from app.services.router_service import RouterService
from app.services.semantic_placeholder import SemanticPlaceholderService


@dataclass
class ServiceContainer:
    """应用服务容器。"""

    settings: Settings
    task_repository: TaskRepository
    template_repository: TemplateRepository
    key_alias_repository: KeyAliasRepository
    ocr_engine: BaseOcrEngine
    vlm_client: OpenAiCompatibleClient
    llm_client: OpenAiCompatibleClient
    semantic_service: SemanticPlaceholderService
    preprocess_service: PreprocessService
    router_service: RouterService
    left_channel_service: LeftChannelService
    right_channel_service: RightChannelService
    fusion_service: FusionService
    key_mapping_service: KeyMappingService
    export_service: ExportService
    pipeline_service: PipelineService


def build_container(settings: Settings | None = None) -> ServiceContainer:
    """构建服务容器。"""

    resolved_settings = settings or get_settings()
    task_repository = TaskRepository(resolved_settings)
    template_repository = TemplateRepository(resolved_settings)
    key_alias_repository = KeyAliasRepository(resolved_settings)
    ocr_engine = build_ocr_engine(resolved_settings)
    vlm_client = OpenAiCompatibleClient(
        base_url=resolved_settings.vlm_base_url,
        api_key=resolved_settings.vlm_api_key,
        timeout_seconds=resolved_settings.vlm_timeout_seconds,
    )
    llm_client = OpenAiCompatibleClient(
        base_url=resolved_settings.llm_base_url,
        api_key=resolved_settings.llm_api_key,
        timeout_seconds=resolved_settings.llm_timeout_seconds,
    )
    semantic_service = SemanticPlaceholderService(resolved_settings)
    preprocess_service = PreprocessService(resolved_settings, task_repository, ocr_engine)
    router_service = RouterService(resolved_settings, template_repository)
    left_channel_service = LeftChannelService(resolved_settings, task_repository, ocr_engine)
    right_channel_service = RightChannelService(resolved_settings, task_repository, vlm_client)
    fusion_service = FusionService()
    key_mapping_service = KeyMappingService(
        resolved_settings,
        template_repository,
        key_alias_repository,
        llm_client,
    )
    export_service = ExportService(resolved_settings, task_repository)
    pipeline_service = PipelineService(
        repository=task_repository,
        preprocess_service=preprocess_service,
        router_service=router_service,
        left_channel_service=left_channel_service,
        right_channel_service=right_channel_service,
        fusion_service=fusion_service,
        key_mapping_service=key_mapping_service,
        export_service=export_service,
        max_upload_mb=resolved_settings.max_upload_mb,
    )
    return ServiceContainer(
        settings=resolved_settings,
        task_repository=task_repository,
        template_repository=template_repository,
        key_alias_repository=key_alias_repository,
        ocr_engine=ocr_engine,
        vlm_client=vlm_client,
        llm_client=llm_client,
        semantic_service=semantic_service,
        preprocess_service=preprocess_service,
        router_service=router_service,
        left_channel_service=left_channel_service,
        right_channel_service=right_channel_service,
        fusion_service=fusion_service,
        key_mapping_service=key_mapping_service,
        export_service=export_service,
        pipeline_service=pipeline_service,
    )
