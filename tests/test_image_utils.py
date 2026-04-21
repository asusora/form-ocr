"""图像坐标工具测试。"""

from __future__ import annotations

from app.models.common import BBox
from app.utils.image_utils import bbox_center, map_bbox_to_original_space, rotate_bbox


def test_map_bbox_to_original_space_preserves_center_after_inverse_rotation() -> None:
    """逆旋转映射后，字段框中心应回到原图对应位置。"""

    width = 1200
    height = 1600
    rotation_degree = 4.5
    origin_bbox = BBox(x=0.24, y=0.31, w=0.28, h=0.07)

    preprocessed_bbox = rotate_bbox(origin_bbox, width, height, rotation_degree)
    recovered_bbox = map_bbox_to_original_space(preprocessed_bbox, width, height, rotation_degree)

    origin_center_x, origin_center_y = bbox_center(origin_bbox)
    recovered_center_x, recovered_center_y = bbox_center(recovered_bbox)

    assert abs(origin_center_x - recovered_center_x) < 1e-3
    assert abs(origin_center_y - recovered_center_y) < 1e-3
    assert recovered_bbox.x <= origin_bbox.x
    assert recovered_bbox.y <= origin_bbox.y
    assert recovered_bbox.x + recovered_bbox.w >= origin_bbox.x + origin_bbox.w
    assert recovered_bbox.y + recovered_bbox.h >= origin_bbox.y + origin_bbox.h
