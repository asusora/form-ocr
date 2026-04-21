"""路由服务测试。"""

from pathlib import Path

from app.core.config import Settings
from app.models.common import AnchorText, BBox, PageAsset
from app.models.stages import Stage0Output
from app.repositories.template_repository import TemplateRepository
from app.services.router_service import RouterService


def build_settings(tmp_path: Path) -> Settings:
    """创建测试配置。"""

    return Settings(
        FORM_OCR_DATA_DIR=str(tmp_path / "artifacts"),
        FORM_OCR_TEMPLATES_PATH=str(tmp_path / "template_registry.json"),
        FORM_OCR_OCR_ENGINE="noop",
    )


def test_router_returns_template_hit_when_anchor_matches(tmp_path: Path) -> None:
    """当锚点一致时应命中模板。"""

    settings = build_settings(tmp_path)
    settings.templates_registry_path.write_text(
        """
        {
          "templates": [
            {
              "template_id": "tpl_demo",
              "template_revision": 2,
              "page_count": 1,
              "anchors": [
                {
                  "page_index": 0,
                  "text": "Name",
                  "bbox": {
                    "x": 0.10,
                    "y": 0.10,
                    "w": 0.08,
                    "h": 0.02
                  }
                }
              ],
              "fields": [
                {
                  "page_index": 0,
                  "canonical_key_id": "ck_name",
                  "field_type": "text",
                  "bbox_relative": {
                    "x": 0.25,
                    "y": 0.10,
                    "w": 0.25,
                    "h": 0.03
                  },
                  "key": "姓名",
                  "key_en": "Name"
                }
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    repository = TemplateRepository(settings)
    service = RouterService(settings, repository)
    stage0_output = Stage0Output(
        task_id="task_test",
        pages=[
            PageAsset(
                page_id="page_task_test_0000",
                task_id="task_test",
                page_index=0,
                width=1000,
                height=1400,
                dpi=300,
                origin_image_url="/artifacts/tasks/task_test/pages/page_0_origin.png",
                preprocessed_image_url="/artifacts/tasks/task_test/pages/page_0_preprocessed.png",
                rotation_degree=0.0,
                deskew_applied=False,
            )
        ],
        anchors=[
            AnchorText(
                page_id="page_task_test_0000",
                anchor_id="anchor_0",
                text="Name",
                bbox=BBox(x=0.10, y=0.10, w=0.08, h=0.02),
                confidence=0.99,
                language="en",
            )
        ],
    )

    decision = service.decide(stage0_output)

    assert decision.route_type == "template_hit"
    assert decision.matched_template is not None
    assert decision.matched_template.template_id == "tpl_demo"
