"""系统级接口。"""

from fastapi import APIRouter, Request

from app.core.container import ServiceContainer
from app.models.common import HealthResponse, LlmConfigStatus

router = APIRouter(tags=["system"])


def get_container(request: Request) -> ServiceContainer:
    """从应用状态中读取容器。"""

    return request.app.state.container


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """返回服务健康状态。"""

    container = get_container(request)
    return HealthResponse(
        status="ok",
        app_name=container.settings.app_name,
        pipeline_version=container.settings.pipeline_version,
    )


@router.get("/llm-config", response_model=LlmConfigStatus)
def llm_config_status(request: Request) -> LlmConfigStatus:
    """返回本地模型配置状态。"""

    container = get_container(request)
    return container.semantic_service.get_status()
