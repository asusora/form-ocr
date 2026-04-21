"""路由决策与模板匹配实现。"""

from __future__ import annotations

from collections import defaultdict

from app.core.config import Settings
from app.models.common import KnownFieldRegion, MatchedTemplate, PageRoute, RouteType
from app.models.stages import RouteDecision, Stage0Output, TemplateDefinition
from app.repositories.template_repository import TemplateRepository
from app.utils.image_utils import bbox_center
from app.utils.text_utils import jaccard_similarity, normalize_text


class RouterService:
    """根据锚点与模板注册表决定路由。"""

    def __init__(self, settings: Settings, template_repository: TemplateRepository) -> None:
        """注入依赖。"""

        self.settings = settings
        self.template_repository = template_repository

    def decide(self, stage0_output: Stage0Output) -> RouteDecision:
        """执行路由决策。"""

        registry = self.template_repository.load_registry()
        if not registry.templates:
            return self._build_unknown(stage0_output, "模板库为空，直接走完整检测路径。")

        best_template: TemplateDefinition | None = None
        best_score = -1.0
        best_anchor_score = 0.0
        best_layout_score = 0.0
        best_page_score = 0.0

        for template in registry.templates:
            anchor_score = self._anchor_text_score(stage0_output, template)
            layout_score = self._layout_score(stage0_output, template)
            page_score = self._page_count_score(len(stage0_output.pages), template.page_count)
            total = (
                anchor_score * self.settings.template_anchor_weight
                + layout_score * self.settings.template_layout_weight
                + page_score * self.settings.template_page_count_weight
            )
            if total > best_score:
                best_template = template
                best_score = total
                best_anchor_score = anchor_score
                best_layout_score = layout_score
                best_page_score = page_score

        if best_template is None:
            return self._build_unknown(stage0_output, "未找到可用模板，直接走完整检测路径。")

        if best_score >= self.settings.template_hit_threshold:
            route_type = RouteType.TEMPLATE_HIT
        elif best_score >= self.settings.template_partial_threshold:
            route_type = RouteType.TEMPLATE_PARTIAL
        else:
            return self._build_unknown(
                stage0_output,
                (
                    "最佳模板分数不足。"
                    f" anchor={best_anchor_score:.3f},"
                    f" layout={best_layout_score:.3f},"
                    f" page={best_page_score:.3f},"
                    f" total={best_score:.3f}"
                ),
            )

        return RouteDecision(
            task_id=stage0_output.task_id,
            route_type=route_type,
            matched_template=MatchedTemplate(
                template_id=best_template.template_id,
                template_revision=best_template.template_revision,
                confidence=round(best_score, 4),
            ),
            page_routes=self._build_page_routes(stage0_output, best_template, route_type),
            reason=(
                f"模板匹配完成。anchor={best_anchor_score:.3f}, "
                f"layout={best_layout_score:.3f}, page={best_page_score:.3f}, "
                f"total={best_score:.3f}"
            ),
        )

    def _build_unknown(self, stage0_output: Stage0Output, reason: str) -> RouteDecision:
        """构建未知模板路由结果。"""

        return RouteDecision(
            task_id=stage0_output.task_id,
            route_type=RouteType.TEMPLATE_UNKNOWN,
            matched_template=None,
            page_routes=[
                PageRoute(
                    page_index=page.page_index,
                    route_type=RouteType.TEMPLATE_UNKNOWN,
                    unknown_region_strategy="run_full_left_channel",
                )
                for page in stage0_output.pages
            ],
            reason=reason,
        )

    def _build_page_routes(
        self,
        stage0_output: Stage0Output,
        template: TemplateDefinition,
        route_type: RouteType,
    ) -> list[PageRoute]:
        """按页组装路由结果。"""

        fields_by_page: dict[int, list] = defaultdict(list)
        for field in template.fields:
            fields_by_page[field.page_index].append(field)

        routes: list[PageRoute] = []
        for page in stage0_output.pages:
            routes.append(
                PageRoute(
                    page_index=page.page_index,
                    route_type=route_type,
                    template_id=template.template_id,
                    template_revision=template.template_revision,
                    affine_ready=False,
                    known_field_regions=[
                        KnownFieldRegion(
                            canonical_key_id=field.canonical_key_id,
                            field_type=field.field_type,
                            bbox_relative=field.bbox_relative,
                            key=field.key,
                            key_en=field.key_en,
                        )
                        for field in fields_by_page.get(page.page_index, [])
                    ],
                    unknown_region_strategy=(
                        "run_dual_channel_for_unmatched_regions"
                        if route_type == RouteType.TEMPLATE_PARTIAL
                        else "template_seed_with_local_snap"
                    ),
                )
            )
        return routes

    def _anchor_text_score(self, stage0_output: Stage0Output, template: TemplateDefinition) -> float:
        """计算锚点文本重合度。"""

        task_texts = [anchor.text for anchor in stage0_output.anchors]
        template_texts = [anchor.text for anchor in template.anchors]
        return jaccard_similarity(task_texts, template_texts)

    def _page_count_score(self, current_count: int, template_count: int) -> float:
        """计算页数一致性分数。"""

        if current_count == template_count:
            return 1.0
        max_count = max(current_count, template_count)
        if max_count == 0:
            return 0.0
        return max(0.0, 1.0 - abs(current_count - template_count) / max_count)

    def _layout_score(self, stage0_output: Stage0Output, template: TemplateDefinition) -> float:
        """根据同名锚点的相对位置评估布局一致性。"""

        task_anchors: dict[tuple[int, str], tuple[float, float]] = {}
        page_lookup = {page.page_id: page.page_index for page in stage0_output.pages}
        for anchor in stage0_output.anchors:
            page_index = page_lookup.get(anchor.page_id)
            if page_index is None:
                continue
            task_anchors[(page_index, normalize_text(anchor.text))] = bbox_center(anchor.bbox)

        distances: list[float] = []
        for anchor in template.anchors:
            normalized_text = normalize_text(anchor.text)
            if not normalized_text:
                continue
            task_point = task_anchors.get((anchor.page_index, normalized_text))
            if task_point is None:
                continue
            template_point = bbox_center(anchor.bbox)
            distance = ((task_point[0] - template_point[0]) ** 2 + (task_point[1] - template_point[1]) ** 2) ** 0.5
            distances.append(distance)

        if not distances:
            return 0.0
        average_distance = sum(distances) / len(distances)
        return max(0.0, 1.0 - min(average_distance / 0.35, 1.0))
