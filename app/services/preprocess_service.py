"""阶段 0：预处理与锚点提取。"""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from app.core.config import Settings
from app.core.errors import FormOcrException
from app.models.common import AnchorText, PageAsset, TaskContext
from app.models.stages import Stage0Output
from app.repositories.task_repository import TaskRepository
from app.services.ocr_service import BaseOcrEngine
from app.utils.id_utils import generate_anchor_id, generate_page_id
from app.utils.image_utils import polygon_to_bbox, preprocess_page
from app.utils.text_utils import infer_language


class PreprocessService:
    """负责页面渲染、图像预处理和锚点 OCR。"""

    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        ocr_engine: BaseOcrEngine,
    ) -> None:
        """注入依赖。"""

        self.settings = settings
        self.repository = repository
        self.ocr_engine = ocr_engine

    def run(self, task_context: TaskContext) -> Stage0Output:
        """执行阶段 0。"""

        source_path = self.repository.url_to_path(task_context.source_file_url)
        try:
            page_images = self._render_pages(source_path, task_context.source_file_type)
        except Exception as exc:
            raise FormOcrException(
                code="PREPROCESS_RENDER_FAILED",
                message=f"页面渲染失败: {exc}",
                status_code=500,
            ) from exc

        if not page_images:
            raise FormOcrException(
                code="PREPROCESS_RENDER_FAILED",
                message="未生成任何页面图像。",
                status_code=500,
            )

        pages: list[PageAsset] = []
        anchors: list[AnchorText] = []
        for page_index, page_image in enumerate(page_images):
            page_id = generate_page_id(task_context.task_id, page_index)
            origin_path = self.repository.build_artifact_path(
                task_context.task_id,
                "pages",
                f"page_{page_index}_origin.png",
            )
            cv2.imwrite(str(origin_path), page_image)

            preprocessed_image, rotation_degree, deskew_applied = preprocess_page(page_image)
            preprocessed_path = self.repository.build_artifact_path(
                task_context.task_id,
                "pages",
                f"page_{page_index}_preprocessed.png",
            )
            cv2.imwrite(str(preprocessed_path), preprocessed_image)

            height, width = preprocessed_image.shape[:2]
            pages.append(
                PageAsset(
                    page_id=page_id,
                    task_id=task_context.task_id,
                    page_index=page_index,
                    width=width,
                    height=height,
                    dpi=self.settings.render_dpi,
                    origin_image_url=self.repository.path_to_url(origin_path),
                    preprocessed_image_url=self.repository.path_to_url(preprocessed_path),
                    rotation_degree=rotation_degree,
                    deskew_applied=deskew_applied,
                )
            )
            anchors.extend(self._extract_anchors(preprocessed_image, page_id, width, height))

        return Stage0Output(
            task_id=task_context.task_id,
            pages=pages,
            anchors=anchors,
        )

    def _render_pages(self, source_path: Path, content_type: str) -> list[np.ndarray]:
        """将 PDF 或图片渲染为页面数组。"""

        suffix = source_path.suffix.lower()
        if content_type == "application/pdf" or suffix == ".pdf":
            zoom = self.settings.render_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pages: list[np.ndarray] = []
            with fitz.open(str(source_path)) as document:
                for page in document:
                    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                        pixmap.height,
                        pixmap.width,
                        pixmap.n,
                    )
                    pages.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return pages

        if suffix in {".png", ".jpg", ".jpeg"} or content_type.startswith("image/"):
            image = Image.open(source_path).convert("RGB")
            return [cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)]

        raise FormOcrException(
            code="INGESTION_FILE_UNSUPPORTED",
            message=f"不支持的文件格式: {source_path.suffix}",
            status_code=400,
        )

    def _extract_anchors(
        self,
        image: np.ndarray,
        page_id: str,
        width: int,
        height: int,
    ) -> list[AnchorText]:
        """提取整页锚点文本。"""

        anchors: list[AnchorText] = []
        for index, line in enumerate(self.ocr_engine.read(image)):
            text = (line.text or "").strip()
            if not text:
                continue
            if line.score < self.settings.anchor_confidence_threshold:
                continue
            bbox = polygon_to_bbox(line.polygon, width, height)
            anchors.append(
                AnchorText(
                    page_id=page_id,
                    anchor_id=generate_anchor_id(page_id, index),
                    text=text,
                    bbox=bbox,
                    confidence=float(line.score),
                    language=infer_language(text),
                )
            )
        return anchors
