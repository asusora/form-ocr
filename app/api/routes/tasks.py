"""任务与 M1 流程接口。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Request, UploadFile, status

from app.core.container import ServiceContainer
from app.models.common import TaskContext
from app.models.stages import ExportOutput

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_container(request: Request) -> ServiceContainer:
    """从应用状态中读取容器。"""

    return request.app.state.container


@router.post("", response_model=TaskContext, status_code=status.HTTP_201_CREATED)
async def create_task(request: Request, file: UploadFile = File(...)) -> TaskContext:
    """上传源文件并创建任务。"""

    container = get_container(request)
    file_bytes = await file.read()
    return container.pipeline_service.create_task(
        file_name=file.filename or "uploaded.bin",
        content_type=file.content_type or "application/octet-stream",
        file_bytes=file_bytes,
    )


@router.get("/{task_id}", response_model=TaskContext)
def get_task(task_id: str, request: Request) -> TaskContext:
    """查询任务信息。"""

    container = get_container(request)
    return container.task_repository.get_task_context(task_id)


@router.post("/{task_id}/run-m1", response_model=ExportOutput)
def run_m1(task_id: str, request: Request) -> ExportOutput:
    """执行 M1 基础识别闭环。"""

    container = get_container(request)
    return container.pipeline_service.run_m1(task_id)


@router.get("/{task_id}/results/export", response_model=ExportOutput)
def get_export_result(task_id: str, request: Request) -> ExportOutput:
    """读取 M1 导出结果。"""

    container = get_container(request)
    stage_output = container.task_repository.load_stage_output(task_id, "export_output")
    return ExportOutput.model_validate(stage_output)


@router.get("/{task_id}/debug/{stage_name}")
def get_stage_output(task_id: str, stage_name: str, request: Request) -> dict[str, Any]:
    """读取阶段调试结果。"""

    container = get_container(request)
    return container.task_repository.load_stage_output(task_id, stage_name)
