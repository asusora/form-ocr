"""统一错误定义。"""

from dataclasses import dataclass


@dataclass
class FormOcrException(Exception):
    """业务异常对象。"""

    code: str
    message: str
    status_code: int = 400

    def __str__(self) -> str:
        """返回可读错误信息。"""

        return f"{self.code}: {self.message}"
