"""API 冒烟测试。"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import Settings
from app.main import create_app


def build_settings(tmp_path: Path) -> Settings:
    """创建测试配置。"""

    return Settings(
        FORM_OCR_DATA_DIR=str(tmp_path / "artifacts"),
        FORM_OCR_TEMPLATES_PATH=str(tmp_path / "template_registry.json"),
        FORM_OCR_OCR_ENGINE="noop",
    )


def build_form_image_bytes() -> bytes:
    """构造一个简单表单图像。"""

    canvas = np.full((600, 900, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Name", (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(canvas, (220, 130), (720, 130), (0, 0, 0), 3)
    cv2.rectangle(canvas, (70, 240), (110, 280), (0, 0, 0), 2)
    cv2.line(canvas, (78, 258), (92, 272), (0, 0, 0), 3)
    cv2.line(canvas, (92, 272), (106, 246), (0, 0, 0), 3)
    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_m1_pipeline_smoke(tmp_path: Path) -> None:
    """上传图片后应能完成 M1 并返回导出结果。"""

    app = create_app(build_settings(tmp_path))
    client = TestClient(app)

    upload_response = client.post(
        "/api/v1/tasks",
        files={"file": ("sample_form.png", build_form_image_bytes(), "image/png")},
    )
    assert upload_response.status_code == 201
    task_id = upload_response.json()["task_id"]

    run_response = client.post(f"/api/v1/tasks/{task_id}/run-m1")
    assert run_response.status_code == 200
    payload = run_response.json()

    assert payload["task_status"] == "exported"
    assert payload["pages"][0]["page_index"] == 0
    assert len(payload["fields"]) >= 1
    first_field = payload["fields"][0]
    assert "origin_bbox" in first_field
    assert "crop_bbox" in first_field
    assert first_field["crop_bbox"]["w"] >= first_field["bbox"]["w"]
    assert first_field["crop_bbox"]["h"] >= first_field["bbox"]["h"]
    line_top = 130 / 600
    line_bottom = 133 / 600
    top_extension = line_top - first_field["bbox"]["y"]
    bottom_extension = (first_field["bbox"]["y"] + first_field["bbox"]["h"]) - line_bottom
    assert top_extension > bottom_extension
