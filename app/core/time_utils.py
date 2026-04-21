"""时间工具函数。"""

from datetime import datetime
from zoneinfo import ZoneInfo


SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    """返回上海时区的 ISO 时间字符串。"""

    return datetime.now(tz=SHANGHAI_TZ).isoformat()
