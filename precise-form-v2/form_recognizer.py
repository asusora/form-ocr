"""
routers/form_recognizer.py

Precise Forms V2 字段自动检测端点。
"""
from __future__ import annotations

import base64
import re
from typing import Annotated, Literal

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import field_detector
import value_extractor
from dependencies import get_jwt_token
from ocr_engine import EasyOCREngine

router = APIRouter(
    prefix="/api/ai/form-recognizer",
    tags=["form-recognizer"],
    responses={404: {"description": "Not found"}},
)

_ocr_engine: "EasyOCREngine | None" = None


def _get_ocr_engine() -> EasyOCREngine:
    """
    获取 OCR 引擎单例，避免每次请求重复初始化。
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = EasyOCREngine(gpu=True)
    return _ocr_engine


class DetectFieldsRequest(BaseModel):
    """
    单页字段检测请求。
    """

    image_base64: str
    page: int = 1


class DetectedFieldResponse(BaseModel):
    """
    单个检测字段的响应数据。
    """

    id: int
    suggested_key: str
    bbox: list[float]
    field_type: str
    confidence: Literal["high", "low"]


class DetectFieldsResponse(BaseModel):
    """
    单页字段检测响应。
    """

    page: int
    fields: list[DetectedFieldResponse]


class DetectFieldsBatchRequest(BaseModel):
    """
    多页字段检测请求。
    """

    pages: list[DetectFieldsRequest] = Field(default_factory=list)


class DetectFieldsBatchPageResponse(BaseModel):
    """
    多页检测中单页的结果响应。
    """

    page: int
    response_code: int
    fields: list[DetectedFieldResponse] = Field(default_factory=list)
    error: str | None = None


class DetectFieldsBatchResponse(BaseModel):
    """
    多页字段检测响应。
    """

    pages: list[DetectFieldsBatchPageResponse] = Field(default_factory=list)


def _base64_to_bgr(image_base64: str) -> np.ndarray:
    """
    将 base64 图片解码为 BGR numpy 数组。

    支持普通 base64 与 data URI 两种格式。
    """
    b64_data = re.sub(r"^data:image/[^;]+;base64,", "", image_base64)
    try:
        img_bytes = base64.b64decode(b64_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image data") from exc

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    return img


def _detect_fields(body: DetectFieldsRequest) -> DetectFieldsResponse:
    """
    执行单页字段检测与 OCR key 建议提取。
    """
    image = _base64_to_bgr(body.image_base64)

    try:
        detected = field_detector.detect_fields(image)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Field detection failed: {str(exc)}") from exc

    if not detected:
        return DetectFieldsResponse(page=body.page, fields=[])

    fields_for_ocr = [
        {
            "id": index + 1,
            "key": "",
            "bbox": list(field.bbox),
            "field_type": field.field_type,
            "confidence": field.confidence,
        }
        for index, field in enumerate(detected)
    ]

    ocr_engine = _get_ocr_engine()
    try:
        labeled = value_extractor.extract_values(image, fields_for_ocr, ocr_engine)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(exc)}") from exc

    result_fields = [
        DetectedFieldResponse(
            id=field["id"],
            suggested_key=field.get("value", "").strip(),
            bbox=field["bbox"],
            field_type=field["field_type"],
            confidence=field["confidence"],
        )
        for field in labeled
    ]

    return DetectFieldsResponse(page=body.page, fields=result_fields)


@router.post("/detect", response_model=DetectFieldsResponse)
def detect_fields_endpoint(
    body: DetectFieldsRequest,
    token: Annotated[dict, Depends(get_jwt_token)],
):
    """
    Precise Forms V2 单页字段检测接口。
    """
    return _detect_fields(body)


@router.post("/detect/batch", response_model=DetectFieldsBatchResponse)
def detect_fields_batch_endpoint(
    body: DetectFieldsBatchRequest,
    token: Annotated[dict, Depends(get_jwt_token)],
):
    """
    Precise Forms V2 多页字段检测接口。

    逐页复用单页检测逻辑，并返回每页 response_code，
    以便前端识别多页场景中的成功页和失败页。
    """
    page_results: list[DetectFieldsBatchPageResponse] = []

    for page_request in body.pages:
        try:
            single_result = _detect_fields(page_request)
            page_results.append(
                DetectFieldsBatchPageResponse(
                    page=single_result.page,
                    response_code=200,
                    fields=single_result.fields,
                )
            )
        except HTTPException as exc:
            page_results.append(
                DetectFieldsBatchPageResponse(
                    page=page_request.page,
                    response_code=exc.status_code,
                    fields=[],
                    error=str(exc.detail),
                )
            )
        except Exception as exc:
            page_results.append(
                DetectFieldsBatchPageResponse(
                    page=page_request.page,
                    response_code=500,
                    fields=[],
                    error=str(exc),
                )
            )

    return DetectFieldsBatchResponse(pages=page_results)
