"""FastAPI 应用入口。"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.system import router as system_router
from app.api.routes.tasks import router as tasks_router
from app.core.config import Settings, get_settings
from app.core.container import build_container
from app.core.errors import FormOcrException


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建 FastAPI 应用实例。"""

    resolved_settings = settings or get_settings()
    container = build_container(resolved_settings)

    app = FastAPI(title=resolved_settings.app_name, version=resolved_settings.pipeline_version)
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_origin_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(FormOcrException)
    async def form_ocr_exception_handler(_: Request, exc: FormOcrException) -> JSONResponse:
        """统一返回业务异常。"""

        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message},
        )

    app.mount(
        "/artifacts",
        StaticFiles(directory=str(resolved_settings.artifacts_root)),
        name="artifacts",
    )
    app.include_router(system_router, prefix=resolved_settings.api_v1_prefix)
    app.include_router(tasks_router, prefix=resolved_settings.api_v1_prefix)
    return app


app = create_app()
