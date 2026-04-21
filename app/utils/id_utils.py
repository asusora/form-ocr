"""标识符生成工具。"""

from datetime import datetime
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def generate_task_id() -> str:
    """生成任务标识。"""

    timestamp = datetime.now(tz=SHANGHAI_TZ).strftime("%Y%m%d_%H%M%S_%f")
    return f"task_{timestamp}"


def generate_page_id(task_id: str, page_index: int) -> str:
    """生成页面标识。"""

    return f"page_{task_id}_{page_index:04d}"


def generate_anchor_id(page_id: str, anchor_index: int) -> str:
    """生成锚点标识。"""

    return f"anchor_{page_id}_{anchor_index:04d}"


def generate_field_id(task_id: str, page_index: int, field_index: int) -> str:
    """生成字段标识。"""

    return f"field_{task_id}_{page_index:04d}_{field_index:04d}"
