"""OCR 适配层。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.core.config import Settings


@dataclass
class OcrLine:
    """统一 OCR 结果行。"""

    polygon: list[list[float]]
    text: str
    score: float


class BaseOcrEngine(ABC):
    """OCR 引擎抽象类。"""

    name: str = "unknown"

    @abstractmethod
    def read(self, image: np.ndarray) -> list[OcrLine]:
        """读取图像中的文字行。"""

    def read_text(self, image: np.ndarray) -> tuple[str, float]:
        """读取完整文本与平均置信度。"""

        lines = self.read(image)
        if not lines:
            return "", 0.0
        lines.sort(key=lambda item: (min(point[1] for point in item.polygon), min(point[0] for point in item.polygon)))
        text = "\n".join(line.text.strip() for line in lines if line.text.strip())
        confidence = sum(line.score for line in lines) / len(lines)
        return text.strip(), float(confidence)


class NoopOcrEngine(BaseOcrEngine):
    """空 OCR 引擎，用于依赖缺失时的降级。"""

    name = "noop"

    def read(self, image: np.ndarray) -> list[OcrLine]:
        """返回空结果。"""

        return []


class RapidOcrEngine(BaseOcrEngine):
    """RapidOCR 引擎包装。"""

    name = "rapidocr"

    def __init__(self) -> None:
        """初始化 RapidOCR。"""

        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> list[OcrLine]:
        """执行 OCR 并归一化输出。"""

        results, _ = self._engine(image)
        normalized: list[OcrLine] = []
        if not results:
            return normalized

        for item in results:
            polygon, text, score = item
            normalized.append(
                OcrLine(
                    polygon=[[float(x), float(y)] for x, y in polygon],
                    text=str(text),
                    score=float(score),
                )
            )
        return normalized


class PaddleOcrEngine(BaseOcrEngine):
    """PaddleOCR 引擎包装。"""

    name = "paddleocr"

    def __init__(self) -> None:
        """初始化 PaddleOCR。"""

        from paddleocr import PaddleOCR

        self._engine = PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)

    def read(self, image: np.ndarray) -> list[OcrLine]:
        """执行 OCR 并归一化输出。"""

        results = self._engine.ocr(image, cls=False)
        normalized: list[OcrLine] = []
        if not results:
            return normalized

        for line in results[0]:
            polygon, content = line
            text, score = content
            normalized.append(
                OcrLine(
                    polygon=[[float(x), float(y)] for x, y in polygon],
                    text=str(text),
                    score=float(score),
                )
            )
        return normalized


def build_ocr_engine(settings: Settings) -> BaseOcrEngine:
    """根据配置创建 OCR 引擎。"""

    engine_name = settings.ocr_engine.lower().strip()
    if engine_name == "paddleocr":
        try:
            return PaddleOcrEngine()
        except Exception:
            return NoopOcrEngine()
    if engine_name == "rapidocr":
        try:
            return RapidOcrEngine()
        except Exception:
            return NoopOcrEngine()
    return NoopOcrEngine()
