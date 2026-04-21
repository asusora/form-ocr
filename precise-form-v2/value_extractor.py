"""
value_extractor.py

Stage 3 of the Form Recognizer pipeline.

Given a page image and a list of fields (each with a key name and bbox),
runs OCR on every bbox region and produces a key-value JSON result.

Input  (from key_assigner / key_assigner_ocr)
-----
  page_image : BGR numpy array
  fields     : [{"id", "key", "bbox", "field_type", "confidence"}, ...]

Output
------
{
  "source": "3000249613.pdf",
  "pages": [
    {
      "page": 1,
      "fields": [
        {
          "id":         1,
          "key":        "申請人姓名",
          "value":      "趙德安",
          "bbox":       [x1, y1, x2, y2],
          "field_type": "underline",
          "confidence": "high"
        },
        ...
      ]
    }
  ]
}

The OCR engine is injected as a dependency (ocr_engine.OCREngine subclass),
so it can be swapped without touching this file.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ocr_engine import OCREngine, EasyOCREngine


# ---------------------------------------------------------------------------
# Slash option value parsing
# ---------------------------------------------------------------------------

# Characters that indicate a crossed-out / eliminated option
_CROSS_PAT = re.compile(r"[Xx×✗✘﹣－\-]{1,2}")


def _parse_selected_option(ocr_text: str, options: list[str]) -> str:
    """
    Given the raw OCR text of a slash_option region and the list of option
    strings, determine which option the user selected.

    Strategy: the user crosses out NON-selected options with "X" or "--".
    An option is considered eliminated if a cross mark appears immediately
    before or after it in the OCR result.

    Returns the selected option string, or the raw OCR text when ambiguous.
    """
    eliminated: set[str] = set()
    for opt in options:
        escaped = re.escape(opt)
        pat = re.compile(
            rf"(?:{_CROSS_PAT.pattern}\s*{escaped}|{escaped}\s*{_CROSS_PAT.pattern})",
            re.IGNORECASE,
        )
        if pat.search(ocr_text):
            eliminated.add(opt)

    remaining = [o for o in options if o not in eliminated]
    if len(remaining) == 1:
        return remaining[0]
    # Ambiguous or all eliminated — return raw OCR
    return ocr_text


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

def extract_values(
    page_image: np.ndarray,
    fields: list[dict],
    ocr_engine: OCREngine,
    page_num: int = 1,
) -> list[dict]:
    """
    Run OCR on each field's bbox and return the fields with a "value" key added.

    Parameters
    ----------
    page_image : BGR numpy array of the page
    fields     : list of field dicts (must contain "bbox"; "key" is optional)
    ocr_engine : OCREngine instance to use for reading
    page_num   : 1-based page number (for logging)

    Returns
    -------
    Copy of fields list, each dict extended with:
        "value" : str  — OCR result for that bbox (empty string if blank/unreadable)
    """
    if not fields:
        return []

    result = []
    for f in fields:
        if f.get("field_type") == "slash_option":
            raw   = ocr_engine.read_region(page_image, f["bbox"])
            value = _parse_selected_option(raw, f.get("options", []))
        else:
            value = ocr_engine.read_region(page_image, f["bbox"])
        out = dict(f)
        out["value"] = value.strip()
        result.append(out)

    return result


# ---------------------------------------------------------------------------
# Multi-page orchestration
# ---------------------------------------------------------------------------

def process_pdf(
    source_name: str,
    page_images: list[np.ndarray],
    page_fields_list: list[list[dict]],
    ocr_engine: Optional[OCREngine] = None,
) -> dict:
    """
    Run value extraction for every page and return the final output dict.

    Parameters
    ----------
    source_name      : original filename (goes into "source" field)
    page_images      : list of BGR page images (one per page)
    page_fields_list : list of field lists (one per page), each field dict
                       must have "bbox" and ideally "key"
    ocr_engine       : OCREngine to use; EasyOCREngine created if None

    Returns
    -------
    {
      "source": "...",
      "pages": [{"page": N, "fields": [...]}]
    }
    """
    if ocr_engine is None:
        ocr_engine = EasyOCREngine()

    pages = []
    for page_idx, (image, fields) in enumerate(zip(page_images, page_fields_list)):
        page_num = page_idx + 1
        print(f"  [value_extractor] Page {page_num}: "
              f"extracting values for {len(fields)} fields …")

        labeled = extract_values(image, fields, ocr_engine, page_num)

        pages.append({
            "page":   page_num,
            "fields": [
                {
                    "id":         f["id"],
                    "key":        f.get("key", ""),
                    "value":      f["value"],
                    "bbox":       f["bbox"],
                    "field_type": f["field_type"],
                    "confidence": f["confidence"],
                }
                for f in labeled
            ],
        })

    return {"source": source_name, "pages": pages}
