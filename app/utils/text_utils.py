"""文本处理工具。"""

import hashlib
import re
from typing import Iterable


WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^0-9a-z\u4e00-\u9fff]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    """对文本执行归一化，便于模板匹配。"""

    compact = WHITESPACE_RE.sub("", text or "").strip().lower()
    return NON_WORD_RE.sub("", compact)


def infer_language(text: str) -> str:
    """根据字符内容推断语言。"""

    if re.search(r"[\u4e00-\u9fff]", text or "") and re.search(r"[A-Za-z]", text or ""):
        return "mixed"
    if re.search(r"[\u4e00-\u9fff]", text or ""):
        return "zh"
    if re.search(r"[A-Za-z]", text or ""):
        return "en"
    return "unknown"


def make_raw_key(text: str, fallback: str) -> str:
    """生成原始展示键。"""

    value = (text or "").strip()
    return value if value else fallback


def build_kv_key(key_text: str, fallback: str, used_keys: set[str]) -> str:
    """生成稳定且去重后的 KV 主键。"""

    normalized = normalize_text(key_text)
    if not normalized:
        digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:10]
        normalized = f"raw_{digest}"
    else:
        normalized = f"raw_{normalized}"

    candidate = normalized
    suffix = 2
    while candidate in used_keys:
        candidate = f"{normalized}__{suffix}"
        suffix += 1

    used_keys.add(candidate)
    return candidate


def jaccard_similarity(values_a: Iterable[str], values_b: Iterable[str]) -> float:
    """计算两个文本集合的 Jaccard 相似度。"""

    set_a = {normalize_text(item) for item in values_a if normalize_text(item)}
    set_b = {normalize_text(item) for item in values_b if normalize_text(item)}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
