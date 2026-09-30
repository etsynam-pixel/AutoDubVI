"""Restore unsafe visual text and render auto-approved Vietnamese replacements."""

from __future__ import annotations

import math
import json
import os
import re
import shutil
import statistics
import subprocess
import time
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from backend.tools.visual_text_style import (
    DESIGN_SYSTEM_VERSION,
    PYCAPS_SAFE_ZONE_BOTTOM_RATIO,
    PYCAPS_SAFE_ZONE_MARGIN_RATIO,
    PYCAPS_SAFE_ZONE_TOP_RATIO,
    resolve_design_role,
    style_for_role,
)

DEFAULT_FONT_FAMILY = "Roboto Slab"
DEFAULT_FONT_KEY = "roboto_slab_bold"
DEFAULT_YELLOW_BGR = (90, 212, 255)
DEFAULT_DARK_BGR = (17, 17, 17)
DEFAULT_WHITE_BGR = (255, 255, 255)
HIGHLIGHT_YELLOW_BGR = (0, 214, 255)
DEFAULT_OUTLINE_BGR = (18, 42, 58)
DEFAULT_BACKGROUND_BGR = (0, 0, 0)
MIN_TEXT_BACKGROUND_CONTRAST = 3.0
MIN_TEXT_OUTLINE_CONTRAST = 3.0
BOX_OPACITY = 0.59
ASS_BOX_ALPHA = round(255 * (1.0 - BOX_OPACITY))
RENDER_STYLE_PRESET = DESIGN_SYSTEM_VERSION
FINAL_QC_FRAME_COUNT = 10

FONT_ASSETS = {
    "roboto_slab_bold": ("RobotoSlab-Bold.ttf", None),
    "be_vietnam_pro_semibold": ("BeVietnamPro-SemiBold.ttf", None),
    "anton": ("Anton-Regular.ttf", None),
    "roboto_condensed_bold": ("RobotoCondensed-Bold.ttf", None),
    "montserrat_extra_bold": ("Montserrat.ttf", "ExtraBold"),
    "montserrat_semibold": ("Montserrat.ttf", "SemiBold"),
    "baloo2_bold": ("Baloo2.ttf", "Bold"),
}


class SilentTextRenderError(RuntimeError):
    pass


SILENT_HARD_SUB_CLASSES = {"silent_dialogue", "narrative_text"}


def _is_safe_vietnamese_replacement(item: dict[str, Any]) -> bool:
    """Defence-in-depth for manual, legacy, or externally edited reviews."""
    return bool(
        item.get("render_action") == "replace_with_vietnamese"
        and item.get("classification") in SILENT_HARD_SUB_CLASSES
        and item.get("text_origin") == "post_production_overlay"
        and str(item.get("vietnamese_translation") or "").strip()
    )


def _font_path(font_key: str = DEFAULT_FONT_KEY) -> Path:
    filename, _variation = FONT_ASSETS.get(
        font_key,
        FONT_ASSETS[DEFAULT_FONT_KEY],
    )
    packaged = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "fonts"
        / filename
    )
    candidates = [packaged]
    if font_key == DEFAULT_FONT_KEY:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/RobotoSlab-Bold.ttf"),
                Path("/usr/share/fonts/truetype/roboto/RobotoSlab-Bold.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SilentTextRenderError(
        f"Video font {filename} is missing from backend/assets/fonts."
    )


def _font(
    size: int,
    font_key: str = DEFAULT_FONT_KEY,
) -> ImageFont.ImageFont:
    selected_key = font_key if font_key in FONT_ASSETS else DEFAULT_FONT_KEY
    font = ImageFont.truetype(str(_font_path(selected_key)), size)
    _filename, variation = FONT_ASSETS[selected_key]
    if variation and hasattr(font, "set_variation_by_name"):
        font.set_variation_by_name(variation)
    return font


def _ass_time(milliseconds: int) -> str:
    centiseconds = max(0, round(milliseconds / 10))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _ass_escape(text: str) -> str:
    return (
        text.replace("\\", "＼")
        .replace("{", "｛")
        .replace("}", "｝")
        .replace("\r", " ")
        .replace("\n", r"\N")
    )


def _normalize_render_line_breaks(text: str) -> str:
    """Canonicalize model/UI line breaks before ASS-safe escaping."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("＼N", "＼n", r"\n"):
        normalized = normalized.replace(marker, r"\N")
    return normalized.replace("\n", r"\N")


def _sentence_case_if_all_caps(text: str) -> str:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 5:
        return text
    words = [
        "".join(character for character in token if character.isalpha())
        for token in text.split()
    ]
    words = [word for word in words if word]
    all_caps = (
        any(character.isupper() for character in letters)
        and not any(character.islower() for character in letters)
    )
    title_case_count = sum(
        1
        for word in words
        if word[0].isupper()
        and (len(word) == 1 or not any(char.isupper() for char in word[1:]))
    )
    excessive_title_case = (
        len(words) >= 5
        and title_case_count >= 4
        and title_case_count / len(words) >= 0.5
    )
    if not all_caps and not excessive_title_case:
        return text
    lowered = text.lower()
    for index, character in enumerate(lowered):
        if character.isalpha():
            return lowered[:index] + character.upper() + lowered[index + 1 :]
    return text


def _wrap_at_width(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or font.getlength(candidate) <= width:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def _balanced_lines(
    text: str,
    font: ImageFont.ImageFont,
    width: int,
    *,
    max_lines: int,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    if font.getlength(text) <= width:
        return [text]
    max_lines = max(1, min(max_lines, len(words)))
    for line_count in range(2, max_lines + 1):
        candidates: list[tuple[float, list[str]]] = []
        for breaks in combinations(range(1, len(words)), line_count - 1):
            starts = (0, *breaks)
            ends = (*breaks, len(words))
            lines = [
                " ".join(words[start:end])
                for start, end in zip(starts, ends, strict=True)
            ]
            widths = [float(font.getlength(line)) for line in lines]
            if max(widths, default=0.0) > width:
                continue
            word_counts = [
                end - start
                for start, end in zip(starts, ends, strict=True)
            ]
            widest = max(widths, default=0.0)
            imbalance = (
                (widest - min(widths, default=widest)) / max(1.0, widest)
            )
            orphan_penalty = 0.35 if min(word_counts, default=2) == 1 else 0.0
            punctuation_bonus = sum(
                0.18
                for line in lines[:-1]
                if line.rstrip().endswith((",", ";", ":", ".", "!", "?"))
            )
            hanging_words = {
                "các",
                "cho",
                "của",
                "đành",
                "hay",
                "là",
                "một",
                "nên",
                "những",
                "sẽ",
                "thì",
                "tôi",
                "trên",
                "trong",
                "và",
                "với",
            }
            hanging_penalty = sum(
                0.20
                for line in lines[:-1]
                if line.rstrip(" ,;:.!?").split()[-1].casefold()
                in hanging_words
            )
            candidates.append(
                (
                    widest / max(1, width)
                    + imbalance * 0.55
                    + orphan_penalty
                    + hanging_penalty
                    - punctuation_bonus,
                    lines,
                )
            )
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    return _wrap_at_width(text, font, width)


def _rendered_text_height(
    lines: list[str],
    font: ImageFont.ImageFont,
    font_size: int,
) -> int:
    heights = []
    for line in lines:
        left, top, right, bottom = font.getbbox(line or "Ág")
        del left, right
        heights.append(max(1, bottom - top))
    line_gap = max(2, round(font_size * 0.12))
    return sum(heights) + line_gap * max(0, len(lines) - 1)


def _fit_text(
    text: str,
    width: int,
    height: int,
    *,
    preferred_size: int | None = None,
    container: bool = False,
    max_lines: int = 2,
    font_key: str = DEFAULT_FONT_KEY,
) -> tuple[str, int]:
    horizontal_padding = max(
        8,
        round(width * 0.05 if container else height * 0.12),
    )
    vertical_padding = max(5, round(height * (0.06 if container else 0.05)))
    usable_width = max(40, width - horizontal_padding * 2)
    usable_height = max(24, height - vertical_padding * 2)
    inferred_size = round(height * (0.62 if container else 0.70))
    start_size = min(96, max(16, int(preferred_size or inferred_size)))
    forced_lines = (
        [line.strip() for line in text.split(r"\N") if line.strip()]
        if r"\N" in text
        else None
    )
    for size in range(start_size, 15, -2):
        font = _font(size, font_key)
        lines = forced_lines or _balanced_lines(
            text, font, usable_width, max_lines=max_lines
        )
        if len(lines) > max_lines:
            continue
        if max((font.getlength(line) for line in lines), default=0) <= usable_width:
            if _rendered_text_height(lines, font, size) <= usable_height:
                return r"\N".join(_ass_escape(line) for line in lines), size
    # Never escape the whole forced-line string: _ass_escape deliberately
    # converts literal backslashes to full-width glyphs, which would turn an
    # ASS line break into visible "＼N" and create a very wide overflowing row.
    # At the advisory minimum size, prefer additional lines over clipping.
    fallback_font = _font(16, font_key)
    source_lines = forced_lines or [text]
    fallback_lines: list[str] = []
    for source_line in source_lines:
        fallback_lines.extend(
            _wrap_at_width(source_line, fallback_font, usable_width)
        )
    return r"\N".join(
        _ass_escape(line) for line in fallback_lines if line.strip()
    ), 16


def _safe_box(value: object, fallback: list[int]) -> list[int]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) for item in value)
    ):
        x1, x2, y1, y2 = (int(round(item)) for item in value)
        if x2 > x1 and y2 > y1:
            return [x1, x2, y1, y2]
    return [int(item) for item in fallback]


def _event_box(
    decision: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    expand_plain: bool,
) -> tuple[list[int], bool]:
    persistent_render_box = (
        decision.get("persistent_render_bbox")
        if bool(decision.get("persistent_fixed_overlay"))
        and bool(decision.get("render_anchor_for_semantic"))
        and decision.get("text_origin") == "post_production_overlay"
        else None
    )
    base = _safe_box(
        decision.get("render_target_bbox") or persistent_render_box or decision.get("bbox"),
        [0, frame_width, 0, frame_height],
    )
    style = (
        decision.get("visual_style")
        if isinstance(decision.get("visual_style"), dict)
        else {}
    )
    container = (
        style.get("container")
        if isinstance(style.get("container"), dict)
        else {}
    )
    is_container = bool(container.get("present"))
    if is_container:
        box = _safe_box(container.get("bbox"), base)
    elif expand_plain:
        x1, x2, y1, y2 = base
        pad_x = max(6, round((x2 - x1) * 0.18))
        pad_y = max(3, round((y2 - y1) * 0.08))
        box = [x1 - pad_x, x2 + pad_x, y1 - pad_y, y2 + pad_y]
    else:
        box = base
    x1, x2, y1, y2 = box
    x1 = max(0, min(frame_width - 1, x1))
    x2 = max(x1 + 1, min(frame_width, x2))
    y1 = max(0, min(frame_height - 1, y1))
    y2 = max(y1 + 1, min(frame_height, y2))
    return [x1, x2, y1, y2], is_container


def _bgr(value: object) -> tuple[int, int, int] | None:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(item, (int, float)) for item in value[:3])
    ):
        return tuple(
            max(0, min(255, int(round(item)))) for item in value[:3]
        )
    return None


def _ass_colour(bgr: tuple[int, int, int]) -> str:
    blue, green, red = bgr
    return f"&H{blue:02X}{green:02X}{red:02X}&"


def _ass_background_shape(
    *,
    text: str,
    font_size: int,
    position_x: int,
    center_y: int,
    alignment: str,
    padding_x: int,
    padding_y: int,
    radius: int,
    background_bgr: tuple[int, int, int],
    background_opacity: float = BOX_OPACITY,
    font_key: str = DEFAULT_FONT_KEY,
    frame_width: int,
    frame_height: int,
    minimum_width: int = 0,
    minimum_height: int = 0,
) -> str:
    """Build a vector background that cannot collide with text events.

    Using a second BorderStyle=3 subtitle as the translucent background can
    corrupt other active glyph bitmaps on some Kaggle libass builds.  A plain
    ASS vector rectangle is visually equivalent but has no text layout state.
    """
    font = _font(font_size, font_key)
    text_lines = text.split(r"\N")
    text_width = max(
        (int(math.ceil(font.getlength(line))) for line in text_lines),
        default=1,
    )
    text_height = _rendered_text_height(text_lines, font, font_size)
    box_width = max(2, int(minimum_width), text_width + padding_x * 2)
    box_height = max(2, int(minimum_height), text_height + padding_y * 2)
    if alignment == "left":
        left = position_x - padding_x
    else:
        left = round(position_x - box_width / 2)
    top = round(center_y - box_height / 2)
    left = max(0, min(frame_width - box_width, left))
    top = max(0, min(frame_height - box_height, top))
    radius = max(0, min(int(radius), box_width // 2, box_height // 2))
    if radius:
        curve = max(1, round(radius * 0.55228475))
        shape = (
            f"m {radius} 0 "
            f"l {box_width - radius} 0 "
            f"b {box_width - radius + curve} 0 {box_width} {radius - curve} "
            f"{box_width} {radius} "
            f"l {box_width} {box_height - radius} "
            f"b {box_width} {box_height - radius + curve} "
            f"{box_width - radius + curve} {box_height} "
            f"{box_width - radius} {box_height} "
            f"l {radius} {box_height} "
            f"b {radius - curve} {box_height} 0 {box_height - radius + curve} "
            f"0 {box_height - radius} "
            f"l 0 {radius} "
            f"b 0 {radius - curve} {radius - curve} 0 {radius} 0"
        )
    else:
        shape = (
            f"m 0 0 l {box_width} 0 l {box_width} {box_height} "
            f"l 0 {box_height}"
        )
    ass_alpha = round(255 * (1.0 - max(0.0, min(1.0, background_opacity))))
    return (
        rf"{{\an7\pos({left},{top})\p1"
        rf"\1c{_ass_colour(background_bgr)}"
        rf"\1a&H{ass_alpha:02X}&\bord0\shad0}}"
        + shape
        + r"{\p0}"
    )


def _expand_background_box_to_minimum(
    box: list[int] | tuple[int, int, int, int],
    *,
    minimum_width: int,
    minimum_height: int,
    center_x: int,
    center_y: int,
) -> list[int]:
    """Expand an x1/x2/y1/y2 box around the original text centre."""
    x1, x2, y1, y2 = (int(value) for value in box)
    target_width = max(x2 - x1, int(minimum_width))
    target_height = max(y2 - y1, int(minimum_height))
    minimum_x1 = round(center_x - target_width / 2)
    minimum_x2 = minimum_x1 + target_width
    minimum_y1 = round(center_y - target_height / 2)
    minimum_y2 = minimum_y1 + target_height
    return [
        min(x1, minimum_x1),
        max(x2, minimum_x2),
        min(y1, minimum_y1),
        max(y2, minimum_y2),
    ]


def _relative_luminance(bgr: tuple[int, int, int]) -> float:
    channels = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]
    linear = [
        channel / 12.92
        if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return (
        0.2126 * linear[0]
        + 0.7152 * linear[1]
        + 0.0722 * linear[2]
    )


def _contrast_ratio(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
) -> float:
    first_luminance = _relative_luminance(first)
    second_luminance = _relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _mix_colour(
    source: tuple[int, int, int],
    target: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(
        int(round(source[index] * (1.0 - amount) + target[index] * amount))
        for index in range(3)
    )


def _translucent_background_contrast(
    text: tuple[int, int, int],
    background: tuple[int, int, int],
    opacity: float = BOX_OPACITY,
) -> float:
    composites = (
        _mix_colour((0, 0, 0), background, opacity),
        _mix_colour((255, 255, 255), background, opacity),
    )
    return min(_contrast_ratio(text, composite) for composite in composites)


def _nearest_contrasting_background(
    text: tuple[int, int, int],
    suggested: tuple[int, int, int],
    opacity: float = BOX_OPACITY,
) -> tuple[int, int, int] | None:
    if (
        _translucent_background_contrast(text, suggested, opacity)
        >= MIN_TEXT_BACKGROUND_CONTRAST
    ):
        return suggested
    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for target in ((0, 0, 0), (255, 255, 255)):
        for step in range(1, 21):
            amount = step / 20
            candidate = _mix_colour(suggested, target, amount)
            if (
                _translucent_background_contrast(text, candidate, opacity)
                >= MIN_TEXT_BACKGROUND_CONTRAST
            ):
                candidates.append((amount, candidate))
                break
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _event_palette(
    decision: dict[str, Any],
    treatment: dict[str, Any] | None = None,
) -> tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    bool,
]:
    style = decision.get("visual_style")
    style = style if isinstance(style, dict) else {}
    container = style.get("container")
    container = container if isinstance(container, dict) else {}
    backgrounds = _container_background_samples(decision)
    if bool(container.get("present")) and backgrounds:
        dark_contrast = min(
            _contrast_ratio(DEFAULT_DARK_BGR, colour) for colour in backgrounds
        )
        white_contrast = min(
            _contrast_ratio(DEFAULT_WHITE_BGR, colour) for colour in backgrounds
        )
        text_bgr = (
            DEFAULT_DARK_BGR
            if dark_contrast >= white_contrast
            else DEFAULT_WHITE_BGR
        )
        return text_bgr, DEFAULT_OUTLINE_BGR, DEFAULT_BACKGROUND_BGR, False
    if bool(container.get("present")) and not treatment:
        # A legacy/partial container record without a measured colour is not
        # reliable enough to preserve. Keep the proven safety fallback.
        return DEFAULT_YELLOW_BGR, DEFAULT_OUTLINE_BGR, DEFAULT_BACKGROUND_BGR, True
    if treatment:
        text = tuple(treatment["text_bgr"])
        outline = tuple(treatment["outline_bgr"])
        background = tuple(treatment["background_bgr"])
        opacity = float(treatment["background_opacity"])
        safe_background = _nearest_contrasting_background(
            text,
            background,
            opacity,
        )
        if safe_background is not None:
            return text, outline, safe_background, True
    # Legacy/neutral data keeps the established yellow-on-59%-black treatment.
    return DEFAULT_YELLOW_BGR, DEFAULT_OUTLINE_BGR, DEFAULT_BACKGROUND_BGR, True


def _safe_bgr(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        colour = tuple(int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if any(item < 0 or item > 255 for item in colour):
        return None
    return colour


def _container_background_samples(
    decision: dict[str, Any],
) -> list[tuple[int, int, int]]:
    style = decision.get("visual_style")
    style = style if isinstance(style, dict) else {}
    container = style.get("container")
    container = container if isinstance(container, dict) else {}
    result: list[tuple[int, int, int]] = []
    values = container.get("background_samples_bgr")
    if isinstance(values, list):
        for value in values:
            colour = _safe_bgr(value)
            if colour is not None and colour not in result:
                result.append(colour)
    fallback = _safe_bgr(container.get("background_bgr"))
    if fallback is not None and fallback not in result:
        result.append(fallback)
    return result


def _event_outline_width(
    decision: dict[str, Any],
    *,
    text_bgr: tuple[int, int, int],
    font_size: int,
    is_container: bool,
    uses_fallback_background: bool,
) -> int:
    default_width = max(1, min(3, int(round(font_size * 0.045))))
    if not is_container or uses_fallback_background:
        return 0
    backgrounds = _container_background_samples(decision)
    if not backgrounds:
        return default_width
    minimum_contrast = min(
        _contrast_ratio(text_bgr, background) for background in backgrounds
    )
    if text_bgr == DEFAULT_DARK_BGR:
        return 0 if minimum_contrast >= 4.5 else 1
    return 1 if minimum_contrast >= 4.5 else 2


def _is_yellow_background(colour: tuple[int, int, int]) -> bool:
    pixel = np.asarray(colour, dtype=np.uint8).reshape(1, 1, 3)
    hue, saturation, value = (
        int(item) for item in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    )
    return 15 <= hue <= 42 and saturation >= 70 and value >= 100


def _event_highlight_colour(
    decision: dict[str, Any],
    *,
    text_bgr: tuple[int, int, int],
    uses_fallback_background: bool,
) -> tuple[int, int, int] | None:
    terms = decision.get("highlight_terms")
    if not isinstance(terms, list) or not any(str(term).strip() for term in terms):
        return None
    # Plain text already uses the selected warm-yellow treatment, so a yellow
    # keyword would not be a meaningful second colour.
    if uses_fallback_background:
        return None
    backgrounds = _container_background_samples(decision)
    if any(_is_yellow_background(colour) for colour in backgrounds):
        return None
    if text_bgr == HIGHLIGHT_YELLOW_BGR:
        return None
    return HIGHLIGHT_YELLOW_BGR


def _highlight_segments(text: str, terms: object) -> list[tuple[str, bool]]:
    if not text or not isinstance(terms, list):
        return [(text, False)]
    candidates: list[str] = []
    for raw in terms[:2]:
        term = re.sub(r"\s+", " ", str(raw).strip())
        if len(term) < 2:
            continue
        candidates.append(term)
    ranges: list[tuple[int, int]] = []
    occupied = [False] * len(text)
    for candidate in sorted(set(candidates), key=len, reverse=True):
        pattern = re.compile(
            rf"(?<!\w){re.escape(candidate)}(?!\w)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            for index in range(start, end):
                occupied[index] = True
            ranges.append((start, end))
    if not ranges:
        return [(text, False)]
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for start, end in sorted(ranges):
        if start > cursor:
            segments.append((text[cursor:start], False))
        segments.append((text[start:end], True))
        cursor = end
    if cursor < len(text):
        segments.append((text[cursor:], False))
    return segments


def _decision_key(decision: dict[str, Any]) -> str:
    source = str(decision.get("source_text_corrected") or "").strip().casefold()
    translated = str(decision.get("vietnamese_translation") or "").strip().casefold()
    return source or translated


def _decision_box(decision: dict[str, Any]) -> list[int]:
    return _safe_box(
        decision.get("render_target_bbox") or decision.get("bbox"), [0, 1, 0, 1]
    )


def _source_visual_units(text: str) -> float:
    units = 0.0
    for character in str(text or "").strip():
        if character.isspace():
            units += 0.35
        elif unicodedata.east_asian_width(character) in {"W", "F"}:
            units += 1.0
        elif unicodedata.category(character).startswith("P"):
            units += 0.45
        else:
            units += 0.62
    return units


def _time_overlap(first: dict[str, Any], second: dict[str, Any]) -> int:
    return max(
        0,
        min(int(first.get("end_ms") or 0), int(second.get("end_ms") or 0))
        - max(
            int(first.get("start_ms") or 0),
            int(second.get("start_ms") or 0),
        ),
    )


def _semantic_frame_interval(
    decision: dict[str, Any],
) -> tuple[int, int] | None:
    """Return the unexpanded OCR interval when the review preserves it."""
    start = decision.get("semantic_start_frame")
    end = decision.get("semantic_end_frame")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    start_frame, end_frame = int(start), int(end)
    if end_frame < start_frame:
        return None
    return start_frame, end_frame


def _meaningful_render_overlap(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    fps: float | None = None,
) -> bool:
    """Ignore mask-padding handoffs while retaining real simultaneous rows."""
    first_semantic = _semantic_frame_interval(first)
    second_semantic = _semantic_frame_interval(second)
    if first_semantic is not None and second_semantic is not None:
        return bool(
            min(first_semantic[1], second_semantic[1])
            >= max(first_semantic[0], second_semantic[0])
        )
    overlap_ms = _time_overlap(first, second)
    if overlap_ms <= 0:
        return False
    frame_ms = int(math.ceil(1000.0 / fps)) if fps and fps > 0 else 50
    return overlap_ms > frame_ms


def _same_visual_row(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    first_key, second_key = _decision_key(first), _decision_key(second)
    first_compact = "".join(first_key.split())
    second_compact = "".join(second_key.split())
    same_content_family = bool(
        first_compact
        and second_compact
        and (
            first_compact == second_compact
            or first_compact in second_compact
            or second_compact in first_compact
        )
    )
    if not same_content_family:
        return False
    if first.get("classification") != second.get("classification"):
        return False
    first_box, second_box = _decision_box(first), _decision_box(second)
    first_width = first_box[1] - first_box[0]
    second_width = second_box[1] - second_box[0]
    first_height = first_box[3] - first_box[2]
    second_height = second_box[3] - second_box[2]
    first_center = (
        (first_box[0] + first_box[1]) / 2,
        (first_box[2] + first_box[3]) / 2,
    )
    second_center = (
        (second_box[0] + second_box[1]) / 2,
        (second_box[2] + second_box[3]) / 2,
    )
    return (
        abs(first_center[0] - second_center[0])
        <= max(32, frame_width * 0.04, max(first_width, second_width) * 0.25)
        and abs(first_center[1] - second_center[1])
        <= max(20, frame_height * 0.015, max(first_height, second_height) * 0.8)
    )


def _adjacent_visual_rows(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    if _decision_key(first) == _decision_key(second):
        return False
    if first.get("classification") != second.get("classification"):
        return False
    if first.get("classification") not in {"sign_or_ui", "narrative_text"}:
        return False
    if _time_overlap(first, second) < 80:
        return False
    first_box, second_box = _decision_box(first), _decision_box(second)
    first_width = first_box[1] - first_box[0]
    second_width = second_box[1] - second_box[0]
    first_height = first_box[3] - first_box[2]
    second_height = second_box[3] - second_box[2]
    if min(first_width, second_width, first_height, second_height) <= 0:
        return False
    width_ratio = max(first_width, second_width) / min(first_width, second_width)
    height_ratio = max(first_height, second_height) / min(
        first_height, second_height
    )
    first_center_y = (first_box[2] + first_box[3]) / 2
    second_center_y = (second_box[2] + second_box[3]) / 2
    first_center_x = (first_box[0] + first_box[1]) / 2
    second_center_x = (second_box[0] + second_box[1]) / 2
    center_gap = abs(first_center_y - second_center_y)
    upper, lower = (
        (first_box, second_box)
        if first_center_y <= second_center_y
        else (second_box, first_box)
    )
    row_gap = lower[2] - upper[3]
    return (
        # Rows in a centred title often have very different widths, so their
        # left edges do not line up even though they belong to one visual
        # block.  Compare centres as well as left edges; otherwise the longer
        # Vietnamese row may wrap into the neighbouring row and overlap it.
        min(
            abs(first_box[0] - second_box[0]),
            abs(first_center_x - second_center_x),
        )
        <= max(20, frame_width * 0.03)
        and width_ratio <= 1.55
        and height_ratio <= 1.55
        and center_gap >= min(first_height, second_height) * 0.55
        and center_gap <= max(first_height, second_height) * 2.5
        and -max(first_height, second_height) * 0.3
        <= row_gap
        <= max(first_height, second_height) * 1.35
        and center_gap <= frame_height * 0.055
    )


def _typography_profiles(
    decisions: list[dict[str, Any]],
    *,
    width: int,
    height: int,
) -> dict[str, dict[str, Any]]:
    parents = list(range(len(decisions)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, second_index in combinations(range(len(decisions)), 2):
        first, second = decisions[first_index], decisions[second_index]
        if _same_visual_row(
            first,
            second,
            frame_width=width,
            frame_height=height,
        ) or _adjacent_visual_rows(
            first,
            second,
            frame_width=width,
            frame_height=height,
        ):
            union(first_index, second_index)

    components: dict[int, list[dict[str, Any]]] = {}
    for index, decision in enumerate(decisions):
        components.setdefault(find(index), []).append(decision)

    profiles: dict[str, dict[str, Any]] = {}
    for component in components.values():
        boxes = [_decision_box(item) for item in component]
        box_heights = [box[3] - box[2] for box in boxes]
        median_height = float(statistics.median(box_heights))
        row_centers: list[float] = []
        for center in sorted((box[2] + box[3]) / 2 for box in boxes):
            if not row_centers or abs(center - row_centers[-1]) > median_height * 0.55:
                row_centers.append(center)
            else:
                row_centers[-1] = (row_centers[-1] + center) / 2
        multirow = len(row_centers) >= 2
        hints = []
        for item in component:
            style = item.get("visual_style")
            style = style if isinstance(style, dict) else {}
            font_hint = style.get("font_hint")
            font_hint = font_hint if isinstance(font_hint, dict) else {}
            value = font_hint.get("size_px")
            if isinstance(value, (int, float)) and value > 0:
                hints.append(float(value))
        if multirow:
            width_estimates = []
            for item, box in zip(component, boxes, strict=True):
                source_units = _source_visual_units(
                    str(item.get("source_text_corrected") or "")
                )
                if source_units >= 1:
                    width_estimates.append(
                        (box[1] - box[0]) / source_units
                    )
            height_estimate = median_height * 0.74
            canonical_size = round(
                min(
                    height_estimate,
                    statistics.median(width_estimates)
                    if width_estimates
                    else height_estimate,
                )
            )
        elif hints:
            canonical_size = round(statistics.median(hints))
        else:
            canonical_size = round(median_height * 0.70)
        canonical_size = max(16, min(96, canonical_size))

        def style_confidence(item: dict[str, Any]) -> tuple[float, int]:
            style = item.get("visual_style")
            style = style if isinstance(style, dict) else {}
            box = _decision_box(item)
            return (
                float(style.get("confidence") or 0.0),
                max(1, box[1] - box[0]) * max(1, box[3] - box[2]),
            )

        reference = max(component, key=style_confidence)
        anchor_x = round(statistics.median(box[0] for box in boxes))
        profile = {
            "font_size": canonical_size,
            "multirow": multirow,
            "alignment": "left" if multirow else "center",
            "anchor_x": anchor_x,
            "reference": reference,
        }
        for item in component:
            profiles[str(item.get("track_id") or "")] = profile
    return profiles


def _expanded_plain_layout_box(
    decision: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    font_size: int,
    text: str,
    alignment: str,
    anchor_x: int | None,
    font_key: str = DEFAULT_FONT_KEY,
) -> list[int]:
    persistent_render_box = (
        decision.get("persistent_render_bbox")
        if bool(decision.get("persistent_fixed_overlay"))
        and bool(decision.get("render_anchor_for_semantic"))
        and decision.get("text_origin") == "post_production_overlay"
        else None
    )
    if persistent_render_box:
        box = _safe_box(persistent_render_box, _decision_box(decision))
        x1, x2, y1, y2 = box
        safe_x = max(12, round(frame_width * 0.025))
        safe_y = max(8, round(frame_height * 0.015))
        x1 = max(safe_x, min(frame_width - safe_x - 1, x1))
        x2 = max(x1 + 1, min(frame_width - safe_x, x2))
        y1 = max(safe_y, min(frame_height - safe_y - 1, y1))
        y2 = max(y1 + 1, min(frame_height - safe_y, y2))
        return [int(x1), int(x2), int(y1), int(y2)]
    base = _decision_box(decision)
    x1, x2, y1, y2 = base
    base_width, base_height = x2 - x1, y2 - y1
    font = _font(font_size, font_key)
    text_lines = [line for line in text.split(r"\N") if line] or [text]
    text_width = int(
        math.ceil(
            max((font.getlength(line) for line in text_lines), default=1)
        )
    )
    text_height = _rendered_text_height(text_lines, font, font_size)
    pad_x = max(8, round(font_size * 0.22))
    pad_y = max(5, round(font_size * 0.16))
    requested_width = text_width + pad_x * 2 + 4
    maximum_expanded_width = max(
        round(base_width * 1.80),
        round(base_width * 1.36),
    )
    target_width = max(
        round(base_width * 1.36),
        min(requested_width, maximum_expanded_width),
    )
    needs_wrap = requested_width > maximum_expanded_width
    target_height = max(
        round(base_height * 1.12),
        text_height
        + pad_y * 2
        + max(4, round(font_size * (0.20 if len(text_lines) > 1 else 0.12))),
    )
    if needs_wrap:
        target_height = max(target_height, round(base_height * 1.65))
    safe_x = max(12, round(frame_width * 0.025))
    safe_y = max(8, round(frame_height * 0.015))
    target_width = min(target_width, frame_width - safe_x * 2)
    target_height = min(target_height, frame_height - safe_y * 2)
    center_y = (y1 + y2) / 2
    if alignment == "left":
        new_x1 = max(safe_x, int(anchor_x if anchor_x is not None else x1))
        new_x2 = min(frame_width - safe_x, new_x1 + target_width)
    else:
        center_x = (x1 + x2) / 2
        new_x1 = round(center_x - target_width / 2)
        new_x2 = new_x1 + target_width
        if new_x1 < safe_x:
            new_x2 += safe_x - new_x1
            new_x1 = safe_x
        if new_x2 > frame_width - safe_x:
            new_x1 -= new_x2 - (frame_width - safe_x)
            new_x2 = frame_width - safe_x
    new_y1 = round(center_y - target_height / 2)
    new_y2 = new_y1 + target_height
    if new_y1 < safe_y:
        new_y2 += safe_y - new_y1
        new_y1 = safe_y
    if new_y2 > frame_height - safe_y:
        new_y1 -= new_y2 - (frame_height - safe_y)
        new_y2 = frame_height - safe_y
    return [int(new_x1), int(new_x2), int(new_y1), int(new_y2)]


def _widen_cinematic_label_box(
    box: list[int],
    *,
    frame_width: int,
    font_size: int,
    text: str,
    alignment: str,
    font_key: str,
) -> list[int]:
    x1, x2, y1, y2 = box
    font = _font(font_size, font_key)
    text_width = int(math.ceil(font.getlength(text.replace(r"\N", " "))))
    horizontal_padding = max(12, round(font_size * 0.50))
    accent_allowance = max(8, round(font_size * 0.22))
    requested_width = text_width + horizontal_padding * 2 + accent_allowance
    safe_x = max(12, round(frame_width * 0.025))
    target_width = min(
        max(x2 - x1, requested_width),
        frame_width - safe_x * 2,
        round(frame_width * 0.42),
    )
    if alignment == "left":
        new_x1 = max(safe_x, x1)
        new_x2 = min(frame_width - safe_x, new_x1 + target_width)
        if new_x2 - new_x1 < target_width:
            new_x1 = max(safe_x, new_x2 - target_width)
    else:
        center_x = (x1 + x2) / 2
        new_x1 = round(center_x - target_width / 2)
        new_x2 = new_x1 + target_width
        if new_x1 < safe_x:
            new_x2 += safe_x - new_x1
            new_x1 = safe_x
        if new_x2 > frame_width - safe_x:
            new_x1 -= new_x2 - (frame_width - safe_x)
            new_x2 = frame_width - safe_x
    return [int(new_x1), int(new_x2), int(y1), int(y2)]


def _preserve_detected_source_rows(
    decision: dict[str, Any],
    translated_text: str,
    *,
    multirow_zone: bool,
) -> str:
    """Keep OCR-separated source rows as separate Vietnamese design rows."""
    if not multirow_zone or r"\N" in translated_text:
        return translated_text
    source_segments = [
        segment.strip()
        for segment in str(
            decision.get("source_text_corrected") or ""
        ).split()
        if segment.strip()
    ]
    if len(source_segments) < 2:
        return translated_text
    for separator in (",", ";", ":"):
        first, found, remainder = translated_text.partition(separator)
        if found and first.strip() and remainder.strip():
            return f"{first.strip()}{r'\N'}{remainder.strip()}"
    return translated_text


def _same_render_row_family(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    first_decision = first["decision"]
    second_decision = second["decision"]
    if bool(first["is_container"]) or bool(second["is_container"]):
        return False
    if first_decision.get("classification") != second_decision.get(
        "classification"
    ):
        return False
    first_key = "".join(_decision_key(first_decision).split())
    second_key = "".join(_decision_key(second_decision).split())
    if not (
        first_key
        and second_key
        and (
            first_key == second_key
            or first_key in second_key
            or second_key in first_key
        )
    ):
        return False
    first_start = int(first_decision.get("start_ms") or 0)
    first_end = int(first_decision.get("end_ms") or 0)
    second_start = int(second_decision.get("start_ms") or 0)
    second_end = int(second_decision.get("end_ms") or 0)
    timeline_gap = max(
        0,
        max(first_start, second_start) - min(first_end, second_end),
    )
    if timeline_gap > 400:
        return False
    first_box, second_box = (
        _decision_box(first_decision),
        _decision_box(second_decision),
    )
    first_height = first_box[3] - first_box[2]
    second_height = second_box[3] - second_box[2]
    first_center_y = (first_box[2] + first_box[3]) / 2
    second_center_y = (second_box[2] + second_box[3]) / 2
    vertical_overlap = max(
        0,
        min(first_box[3], second_box[3])
        - max(first_box[2], second_box[2]),
    )
    strict_static_match = (
        vertical_overlap >= min(first_height, second_height) * 0.65
        and abs(first_center_y - second_center_y)
        <= max(20, frame_height * 0.015)
        and abs(first_box[0] - second_box[0])
        <= max(20, frame_width * 0.03)
    )
    if strict_static_match:
        return True
    first_semantic = _semantic_frame_interval(first_decision)
    second_semantic = _semantic_frame_interval(second_decision)
    if first_semantic is None or second_semantic is None:
        return False
    semantic_overlap = max(
        0,
        min(first_semantic[1], second_semantic[1])
        - max(first_semantic[0], second_semantic[0])
        + 1,
    )
    semantic_gap = max(
        0,
        max(first_semantic[0], second_semantic[0])
        - min(first_semantic[1], second_semantic[1])
        - 1,
    )
    first_width = first_box[1] - first_box[0]
    second_width = second_box[1] - second_box[0]
    horizontal_overlap = max(
        0,
        min(first_box[1], second_box[1])
        - max(first_box[0], second_box[0]),
    )
    return bool(
        semantic_overlap <= 1
        and semantic_gap <= 3
        and horizontal_overlap >= min(first_width, second_width) * 0.35
        and vertical_overlap >= min(first_height, second_height) * 0.25
    )


def _stabilize_direct_specs(
    specs: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Collapse OCR fragments of one static visual row into one stable item."""
    if len(specs) < 2:
        return specs
    parents = list(range(len(specs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, second_index in combinations(range(len(specs)), 2):
        if _same_render_row_family(
            specs[first_index],
            specs[second_index],
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            union(first_index, second_index)

    components: dict[int, list[dict[str, Any]]] = {}
    for index, spec in enumerate(specs):
        components.setdefault(find(index), []).append(spec)

    stabilized: list[dict[str, Any]] = []
    for component in components.values():
        if len(component) == 1:
            stabilized.append(component[0])
            continue
        representative = max(
            component,
            key=lambda item: (
                int(item["decision"].get("end_ms") or 0)
                - int(item["decision"].get("start_ms") or 0),
                len(str(item["text"])),
            ),
        )
        merged = dict(representative)
        decision = dict(representative["decision"])
        decision["start_ms"] = min(
            int(item["decision"].get("start_ms") or 0)
            for item in component
        )
        decision["end_ms"] = max(
            int(item["decision"].get("end_ms") or 0)
            for item in component
        )
        semantic_intervals = [
            interval
            for item in component
            if (interval := _semantic_frame_interval(item["decision"]))
            is not None
        ]
        if semantic_intervals:
            decision["semantic_start_frame"] = min(
                interval[0] for interval in semantic_intervals
            )
            decision["semantic_end_frame"] = max(
                interval[1] for interval in semantic_intervals
            )
        merged["decision"] = decision
        stabilized.append(merged)
    return sorted(
        stabilized,
        key=lambda item: (
            int(item["decision"].get("start_ms") or 0),
            int(item["center_y"]),
            str(item["track_id"]),
        ),
    )


def _expanded_container_layout_box(
    box: list[int],
    *,
    frame_width: int,
    frame_height: int,
    font_size: int,
    text: str,
    font_key: str = DEFAULT_FONT_KEY,
) -> list[int]:
    x1, x2, y1, y2 = box
    base_width, base_height = x2 - x1, y2 - y1
    safe_x = max(12, round(frame_width * 0.025))
    safe_y = max(8, round(frame_height * 0.015))
    maximum_width = min(
        frame_width - safe_x * 2,
        round(base_width * 1.18),
    )
    font = _font(font_size, font_key)
    target_width = base_width
    lines: list[str] | None = None
    for candidate_width in range(base_width, maximum_width + 1, 4):
        horizontal_padding = max(8, round(candidate_width * 0.05))
        usable_width = candidate_width - horizontal_padding * 2
        candidate_lines = _balanced_lines(
            text,
            font,
            usable_width,
            max_lines=2,
        )
        if (
            len(candidate_lines) <= 2
            and max(
                (font.getlength(line) for line in candidate_lines),
                default=0,
            )
            <= usable_width
        ):
            target_width = candidate_width
            lines = candidate_lines
            break
    if lines is None:
        target_width = maximum_width
        horizontal_padding = max(8, round(target_width * 0.05))
        lines = _balanced_lines(
            text,
            font,
            target_width - horizontal_padding * 2,
            max_lines=2,
        )
    vertical_padding = max(5, round(base_height * 0.06))
    required_height = (
        _rendered_text_height(lines, font, font_size)
        + vertical_padding * 2
        + 4
    )
    target_height = min(
        max(base_height, required_height),
        round(base_height * 1.20),
        frame_height - safe_y * 2,
    )
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    new_x1 = round(center_x - target_width / 2)
    new_x2 = new_x1 + target_width
    new_y1 = round(center_y - target_height / 2)
    new_y2 = new_y1 + target_height
    if new_x1 < safe_x:
        new_x2 += safe_x - new_x1
        new_x1 = safe_x
    if new_x2 > frame_width - safe_x:
        new_x1 -= new_x2 - (frame_width - safe_x)
        new_x2 = frame_width - safe_x
    if new_y1 < safe_y:
        new_y2 += safe_y - new_y1
        new_y1 = safe_y
    if new_y2 > frame_height - safe_y:
        new_y1 -= new_y2 - (frame_height - safe_y)
        new_y2 = frame_height - safe_y
    return [int(new_x1), int(new_x2), int(new_y1), int(new_y2)]


def _direct_render_specs(
    review: dict[str, Any],
    *,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    decisions = [
        item
        for item in review.get("decisions") or []
        if _is_safe_vietnamese_replacement(item)
    ]
    if not decisions:
        return []
    typography = _typography_profiles(decisions, width=width, height=height)
    specs: list[dict[str, Any]] = []
    for decision in decisions:
        profile = typography[str(decision.get("track_id") or "")]
        design_role = resolve_design_role(
            decision,
            frame_width=width,
            frame_height=height,
        )
        treatment = style_for_role(design_role, decision)
        font_key = str(treatment["font_key"])
        (x1, x2, y1, y2), is_container = _event_box(
            decision,
            frame_width=width,
            frame_height=height,
            expand_plain=True,
        )
        preferred_size = max(
            16,
            min(
                96,
                round(
                    int(profile["font_size"])
                    * float(treatment["font_size_multiplier"])
                ),
            ),
        )
        persistent_disclaimer_layout = bool(
            decision.get("persistent_render_bbox")
            and decision.get("persistent_fixed_overlay")
            and decision.get("render_anchor_for_semantic")
            and decision.get("text_origin") == "post_production_overlay"
        )
        if persistent_disclaimer_layout:
            preferred_size = max(preferred_size, max(23, round(height * 0.013)))
        display_text = _normalize_render_line_breaks(
            _sentence_case_if_all_caps(
                str(decision.get("vietnamese_translation") or "")
            )
        )
        if persistent_disclaimer_layout:
            lowered = display_text.lower()
            if "bắt chước" in lowered and "hư cấu" in lowered:
                display_text = r"Không bắt chước\NKịch bản hư cấu"
            elif "bắt chước" in lowered:
                display_text = "Không bắt chước"
            elif "hư cấu" in lowered:
                display_text = "Kịch bản hư cấu"
        if bool(profile["multirow"]):
            is_container = False
        display_text = _preserve_detected_source_rows(
            decision,
            display_text,
            multirow_zone=bool(profile["multirow"]),
        )
        cinematic_character_label = bool(treatment.get("force_single_line"))
        if not is_container:
            x1, x2, y1, y2 = _expanded_plain_layout_box(
                decision,
                frame_width=width,
                frame_height=height,
                font_size=preferred_size,
                text=display_text,
                alignment=str(profile["alignment"]),
                anchor_x=int(profile["anchor_x"]),
                font_key=font_key,
            )
            if cinematic_character_label:
                x1, x2, y1, y2 = _widen_cinematic_label_box(
                    [x1, x2, y1, y2],
                    frame_width=width,
                    font_size=preferred_size,
                    text=display_text,
                    alignment=str(profile["alignment"]),
                    font_key=font_key,
                )
        # The original container remains visible in the restored source frame.
        # Direct rendering must therefore fit inside that exact box. Expanding
        # the layout area made long Vietnamese translations spill beyond the
        # rounded box even though their nominal text metrics still fitted.
        captured_outline_width = (
            (
                profile["reference"].get("visual_style")
                if isinstance(profile["reference"].get("visual_style"), dict)
                else {}
            )
            .get("layout", {})
            .get("outline_width_px")
        )
        fit_inset = 0
        if is_container:
            if (
                isinstance(captured_outline_width, (int, float))
                and captured_outline_width > 0
            ):
                fit_inset = max(4, int(round(captured_outline_width)) + 3)
            else:
                fit_inset = max(4, int(round(preferred_size * 0.08)))
        box_width = max(1, (x2 - x1) - fit_inset * 2)
        box_height = max(1, (y2 - y1) - fit_inset * 2)
        text, font_size = _fit_text(
            display_text,
            box_width,
            box_height,
            preferred_size=preferred_size,
            container=is_container,
            max_lines=(
                1
                if cinematic_character_label
                else (
                    2
                    if r"\N" in display_text
                    else (1 if bool(profile["multirow"]) else 2)
                )
            ),
            font_key=font_key,
        )
        # Typography may be shared across adjacent rows, but colour and
        # background ownership are event-specific.  In particular, a mixed
        # group can contain a measured bright container and a plain row.  Once
        # the group is laid out as plain multi-row text, inheriting the bright
        # container palette would produce dark text on a semantic fallback.
        if is_container:
            palette_decision = decision
            (
                text_bgr,
                outline_bgr,
                background_bgr,
                uses_fallback_background,
            ) = _event_palette(palette_decision, treatment)
        else:
            palette_decision = decision
            plain_palette_decision = dict(decision)
            plain_visual_style = dict(
                decision.get("visual_style")
                if isinstance(decision.get("visual_style"), dict)
                else {}
            )
            plain_visual_style["container"] = {"present": False}
            plain_palette_decision["visual_style"] = plain_visual_style
            (
                text_bgr,
                outline_bgr,
                background_bgr,
                uses_fallback_background,
            ) = _event_palette(plain_palette_decision, treatment)
        reference_style = profile["reference"].get("visual_style")
        reference_style = (
            reference_style if isinstance(reference_style, dict) else {}
        )
        layout = (
            reference_style.get("layout")
            if isinstance(reference_style.get("layout"), dict)
            else {}
        )
        captured_outline_width = layout.get("outline_width_px")
        outline_width = _event_outline_width(
            palette_decision,
            text_bgr=text_bgr,
            font_size=font_size,
            is_container=is_container,
            uses_fallback_background=uses_fallback_background,
        )
        if is_container and not uses_fallback_background:
            highlight_bgr = _event_highlight_colour(
                palette_decision,
                text_bgr=text_bgr,
                uses_fallback_background=uses_fallback_background,
            )
        elif any(str(term).strip() for term in decision.get("highlight_terms") or []):
            candidate_highlight = treatment.get("highlight_bgr")
            highlight_bgr = (
                tuple(candidate_highlight)
                if isinstance(candidate_highlight, (list, tuple))
                and len(candidate_highlight) == 3
                else None
            )
        else:
            highlight_bgr = None
        background_opacity = float(treatment["background_opacity"])
        background_padding_x = max(
            8,
            min(26, int(round(font_size * float(treatment["padding_x_ratio"])))),
        )
        background_padding_y = max(
            5,
            min(16, int(round(font_size * float(treatment["padding_y_ratio"])))),
        )
        background_radius = max(
            3,
            min(22, int(round(font_size * float(treatment["radius_ratio"])))),
        )
        center_y = round((y1 + y2) / 2)
        if profile["alignment"] == "left":
            position_x = x1
        else:
            position_x = round((x1 + x2) / 2)
        specs.append(
            {
                "decision": decision,
                "track_id": str(decision.get("track_id") or ""),
                "typography_group": id(profile),
                "design_role": design_role,
                "design_system": DESIGN_SYSTEM_VERSION,
                "cinematic_character_label": cinematic_character_label,
                "text": text,
                "font_size": font_size,
                "font_key": font_key,
                "font_family": str(treatment["font_family"]),
                "text_bgr": text_bgr,
                "outline_bgr": outline_bgr,
                "background_bgr": background_bgr,
                "background_opacity": background_opacity,
                "outline_width": outline_width,
                "highlight_terms": list(
                    decision.get("highlight_terms") or []
                ),
                "highlight_bgr": highlight_bgr,
                "highlight_outline_width": (
                    max(1, min(2, int(round(font_size * 0.035))))
                    if highlight_bgr is not None
                    else outline_width
                ),
                "background_padding_x": background_padding_x,
                "background_padding_y": background_padding_y,
                "background_radius": background_radius,
                "accent_bar_bgr": (
                    tuple(treatment["accent_bar_bgr"])
                    if isinstance(treatment.get("accent_bar_bgr"), (list, tuple))
                    and len(treatment.get("accent_bar_bgr")) == 3
                    else None
                ),
                "accent_bar_width": max(
                    0,
                    int(round(font_size * float(treatment.get("accent_bar_width_ratio", 0.0)))),
                ),
                "accent_bar_inset": max(
                    0,
                    int(round(font_size * float(treatment.get("accent_bar_inset_ratio", 0.0)))),
                ),
                "accent_bar_vertical_margin_ratio": float(
                    treatment.get("accent_bar_vertical_margin_ratio", 0.16)
                ),
                "position_x": position_x,
                "center_y": center_y,
                "alignment": str(profile["alignment"]),
                "is_container": is_container,
                "uses_fallback_background": uses_fallback_background,
                "draw_background": not is_container or uses_fallback_background,
                "minimum_background_width": int(
                    (decision.get("render_target_min_size") or [0, 0])[0]
                    if len(decision.get("render_target_min_size") or []) == 2
                    else 0
                ),
                "minimum_background_height": int(
                    (decision.get("render_target_min_size") or [0, 0])[1]
                    if len(decision.get("render_target_min_size") or []) == 2
                    else 0
                ),
            }
        )
    group_sizes: dict[int, int] = {}
    for spec in specs:
        group = int(spec["typography_group"])
        group_sizes[group] = min(
            int(spec["font_size"]),
            group_sizes.get(group, int(spec["font_size"])),
        )
    for spec in specs:
        font_size = group_sizes[int(spec["typography_group"])]
        spec["font_size"] = font_size
        spec["outline_width"] = _event_outline_width(
            spec["decision"],
            text_bgr=spec["text_bgr"],
            font_size=font_size,
            is_container=bool(spec["is_container"]),
            uses_fallback_background=bool(spec["uses_fallback_background"]),
        )
        if spec.get("highlight_bgr") is not None:
            spec["highlight_outline_width"] = max(
                1,
                min(2, int(round(font_size * 0.035))),
            )
        treatment = style_for_role(
            str(spec["design_role"]),
            spec.get("decision") if isinstance(spec.get("decision"), dict) else None,
        )
        spec["background_padding_x"] = max(
            8,
            min(
                26,
                int(
                    round(
                        font_size
                        * float(treatment["padding_x_ratio"])
                    )
                ),
            ),
        )
        spec["background_padding_y"] = max(
            5,
            min(
                16,
                int(
                    round(
                        font_size
                        * float(treatment["padding_y_ratio"])
                    )
                ),
            ),
        )
        spec["background_radius"] = max(
            3,
            min(
                22,
                int(
                    round(
                        font_size
                        * float(treatment["radius_ratio"])
                    )
                ),
            ),
        )
        spec["accent_bar_width"] = max(
            0,
            int(round(font_size * float(treatment.get("accent_bar_width_ratio", 0.0)))),
        )
        spec["accent_bar_inset"] = max(
            0,
            int(round(font_size * float(treatment.get("accent_bar_inset_ratio", 0.0)))),
        )
    stabilized = _stabilize_direct_specs(
        specs,
        frame_width=width,
        frame_height=height,
    )
    frame_fitted = _fit_plain_specs_inside_frame(
        stabilized,
        frame_width=width,
        frame_height=height,
    )
    pycaps_fitted = _avoid_pycaps_safe_zone(
        frame_fitted,
        frame_width=width,
        frame_height=height,
    )
    resolved = _resolve_simultaneous_spec_overlaps(
        pycaps_fitted,
        frame_width=width,
        frame_height=height,
    )
    return _coalesce_proven_multiline_specs(
        resolved, frame_width=width, frame_height=height,
    )


def _coalesce_proven_multiline_specs(
    specs: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Lay out a verified three-row editor block on one readable panel."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        decision = spec["decision"]
        group_id = str(decision.get("multiline_render_group") or "")
        if group_id:
            groups.setdefault(group_id, []).append(spec)
    result = [dict(spec) for spec in specs]
    by_track = {str(spec["track_id"]): spec for spec in result}
    for group_id, members in groups.items():
        if len(members) != 3:
            continue
        rows = {
            member["decision"].get("multiline_render_row"): by_track[
                str(member["track_id"])
            ]
            for member in members
        }
        if set(rows) != {0, 1, 2} or str(rows[0]["track_id"]) != group_id:
            continue
        ordered = [rows[index] for index in range(3)]
        if any(
            bool(spec.get("is_container"))
            or not bool(spec.get("draw_background"))
            or not bool(spec["decision"].get("selected_for_inpaint"))
            or spec["decision"].get("render_action") != "replace_with_vietnamese"
            for spec in ordered
        ):
            continue
        timelines = {
            (int(spec["decision"].get("start_ms") or 0),
             int(spec["decision"].get("end_ms") or 0))
            for spec in ordered
        }
        if len(timelines) != 1:
            continue
        source_boxes = [
            [int(value) for value in spec["decision"].get("bbox") or []]
            for spec in ordered
        ]
        if any(len(box) != 4 for box in source_boxes):
            continue
        font_size = max(23, round(frame_height * 0.0125))
        line_step = max(32, round(font_size * 1.45))
        center_y = round((min(box[2] for box in source_boxes)
                          + max(box[3] for box in source_boxes)) / 2)
        position_x = max(40, min(box[0] for box in source_boxes)
                         - round(frame_width * 0.08))
        for row, spec in enumerate(ordered):
            spec["font_size"] = font_size
            spec["position_x"] = position_x
            spec["center_y"] = center_y + (row - 1) * line_step
            spec["alignment"] = "left"
            spec["draw_background"] = row == 0
            if row:
                spec["shared_background_owner"] = group_id
            spec["background_padding_x"] = 12
            spec["background_padding_y"] = 8
            spec["background_radius"] = 8
        text_boxes = [_spec_geometry(spec)["text_box"] for spec in ordered]
        panel = [
            min(box[0] for box in text_boxes) - 12,
            max(box[1] for box in text_boxes) + 12,
            min(box[2] for box in text_boxes) - 8,
            max(box[3] for box in text_boxes) + 8,
        ]
        if (
            panel[0] < 0 or panel[1] > frame_width
            or panel[2] < 0 or panel[3] > frame_height
        ):
            continue
        for spec in ordered:
            spec["background_override_box"] = panel
        panel_index = result.index(ordered[0])
        earliest_index = min(result.index(spec) for spec in ordered)
        if panel_index > earliest_index:
            result.pop(panel_index)
            result.insert(earliest_index, ordered[0])
    return result


def _pil_rgb(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return (bgr[2], bgr[1], bgr[0])


def _prepare_direct_spec_patch(
    spec: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any] | None:
    """Render one static text layer once and retain only its occupied pixels."""
    font_size = int(spec["font_size"])
    font = _font(font_size, str(spec.get("font_key") or DEFAULT_FONT_KEY))
    text_lines = str(spec["text"]).split(r"\N")
    line_gap = max(2, round(font_size * 0.12))
    metrics = []
    for line in text_lines:
        left, top, right, bottom = font.getbbox(line or "Ág")
        metrics.append(
            {
                "text": line,
                "left": left,
                "top": top,
                "width": max(1, right - left),
                "height": max(1, bottom - top),
            }
        )
    text_height = (
        sum(item["height"] for item in metrics)
        + line_gap * max(0, len(metrics) - 1)
    )
    position_x = int(spec["position_x"])
    center_y = int(spec["center_y"])
    overlay = Image.new(
        "RGBA",
        (int(frame_width), int(frame_height)),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)
    cursor_y = round(center_y - text_height / 2)
    placements = []
    for metric in metrics:
        if spec["alignment"] == "left":
            line_left = position_x
        else:
            line_left = round(position_x - metric["width"] / 2)
        placements.append((metric, line_left, cursor_y))
        cursor_y += metric["height"] + line_gap
    outline_width = int(spec["outline_width"])
    padding_x = int(spec["background_padding_x"])
    padding_y = int(spec["background_padding_y"])
    text_left = min(line_left for _metric, line_left, _line_y in placements)
    text_right = max(
        line_left + int(metric["width"])
        for metric, line_left, _line_y in placements
    )
    text_top = min(line_y for _metric, _line_left, line_y in placements)
    text_bottom = max(
        line_y + int(metric["height"])
        for metric, _line_left, line_y in placements
    )
    background_geometry = _expand_background_box_to_minimum(
        [
            text_left - outline_width - padding_x,
            text_right + outline_width + padding_x,
            text_top - outline_width - padding_y,
            text_bottom + outline_width + padding_y,
        ],
        minimum_width=int(spec.get("minimum_background_width") or 0),
        minimum_height=int(spec.get("minimum_background_height") or 0),
        center_x=position_x,
        center_y=center_y,
    )
    if spec.get("background_override_box") is not None:
        background_geometry = [
            int(value) for value in spec["background_override_box"]
        ]
    background_width = min(
        overlay.width,
        background_geometry[1] - background_geometry[0],
    )
    background_height = min(
        overlay.height,
        background_geometry[3] - background_geometry[2],
    )
    background_left = max(
        0,
        min(overlay.width - background_width, background_geometry[0]),
    )
    background_top = max(
        0,
        min(overlay.height - background_height, background_geometry[2]),
    )
    background_box = (
        background_left,
        background_top,
        background_left + background_width,
        background_top + background_height,
    )
    if bool(spec.get("draw_background", True)):
        draw.rounded_rectangle(
            background_box,
            radius=int(spec["background_radius"]),
            fill=(
                *_pil_rgb(spec["background_bgr"]),
                round(255 * float(spec.get("background_opacity", BOX_OPACITY))),
            ),
        )
        accent_bgr = spec.get("accent_bar_bgr")
        accent_width = int(spec.get("accent_bar_width") or 0)
        if (
            isinstance(accent_bgr, (list, tuple))
            and len(accent_bgr) == 3
            and accent_width > 0
        ):
            inset = max(2, int(spec.get("accent_bar_inset") or 0))
            vertical_margin_ratio = max(
                0.08,
                min(0.32, float(spec.get("accent_bar_vertical_margin_ratio", 0.16))),
            )
            accent_left = min(
                background_box[2] - accent_width - 1,
                background_box[0] + inset,
            )
            accent_top = background_box[1] + max(
                2,
                int(round(background_height * vertical_margin_ratio)),
            )
            accent_bottom = background_box[3] - max(
                2,
                int(round(background_height * vertical_margin_ratio)),
            )
            if accent_bottom > accent_top:
                draw.rounded_rectangle(
                    (
                        accent_left,
                        accent_top,
                        accent_left + accent_width,
                        accent_bottom,
                    ),
                    radius=max(1, accent_width // 2),
                    fill=(*_pil_rgb(tuple(accent_bgr)), 255),
                )
    for metric, line_left, line_y in placements:
        cursor_x = float(line_left - metric["left"])
        draw_y = line_y - metric["top"]
        for segment, highlighted in _highlight_segments(
            metric["text"],
            spec.get("highlight_terms"),
        ):
            use_highlight = (
                highlighted and spec.get("highlight_bgr") is not None
            )
            colour = (
                spec["highlight_bgr"] if use_highlight else spec["text_bgr"]
            )
            segment_outline = (
                int(spec["highlight_outline_width"])
                if use_highlight
                else outline_width
            )
            draw.text(
                (round(cursor_x), draw_y),
                segment,
                font=font,
                fill=(*_pil_rgb(colour), 255),
                stroke_width=segment_outline,
                stroke_fill=(*_pil_rgb(spec["outline_bgr"]), 255),
            )
            cursor_x += float(font.getlength(segment))
    bounds = overlay.getbbox()
    if bounds is None:
        return None
    return {
        "bounds": bounds,
        "overlay": overlay.crop(bounds),
        "track_id": str(spec.get("track_id") or ""),
    }


def _draw_prepared_direct_patches(
    frame: Any,
    patches: list[dict[str, Any]],
) -> Any:
    if not patches:
        return frame
    rendered = frame.copy()
    for patch in patches:
        left, top, right, bottom = patch["bounds"]
        region = rendered[top:bottom, left:right]
        base = Image.fromarray(
            cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        ).convert("RGBA")
        composited = Image.alpha_composite(base, patch["overlay"])
        rendered[top:bottom, left:right] = cv2.cvtColor(
            np.asarray(composited.convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )
    return rendered


def _draw_direct_specs(
    frame: Any,
    specs: list[dict[str, Any]],
) -> Any:
    patches = [
        patch
        for spec in specs
        if (
            patch := _prepare_direct_spec_patch(
                spec,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0],
            )
        )
        is not None
    ]
    return _draw_prepared_direct_patches(frame, patches)


def write_ass(
    review: dict[str, Any],
    path: str | Path,
    *,
    width: int,
    height: int,
) -> Path | None:
    decisions = [
        item
        for item in review.get("decisions") or []
        if _is_safe_vietnamese_replacement(item)
    ]
    if not decisions:
        return None
    specs = _direct_render_specs(review, width=width, height=height)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: VisualShape,Roboto Slab,42,&HFFFFFFFF,&HFFFFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: VisualVI,Roboto Slab,42,&H005AD4FF,&H005AD4FF,&H00122A3A,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
    """
    lines = [header.rstrip()]
    for spec in specs:
        decision = spec["decision"]
        text = str(spec["text"])
        font_size = int(spec["font_size"])
        font_family = str(spec.get("font_family") or DEFAULT_FONT_FAMILY)
        text_bgr = spec["text_bgr"]
        outline_bgr = spec["outline_bgr"]
        outline_width = int(spec["outline_width"])
        center_y = int(spec["center_y"])
        position_x = int(spec["position_x"])
        alignment = 4 if spec["alignment"] == "left" else 5
        shared = (
            rf"{{\an{alignment}\pos({position_x},{center_y})"
            rf"\fn{font_family}\fs{font_size}\b1"
        )
        text_override = (
            shared
            + rf"\c{_ass_colour(text_bgr)}"
            + rf"\3c{_ass_colour(outline_bgr)}"
            + rf"\bord{outline_width}\shad0}}"
        )
        rendered_text_parts: list[str] = []
        for line_index, line in enumerate(text.split(r"\N")):
            if line_index:
                rendered_text_parts.append(r"\N")
            for segment, highlighted in _highlight_segments(
                line,
                spec.get("highlight_terms"),
            ):
                if highlighted and spec.get("highlight_bgr") is not None:
                    rendered_text_parts.append(
                        "{"
                        + rf"\c{_ass_colour(spec['highlight_bgr'])}"
                        + rf"\3c{_ass_colour(outline_bgr)}"
                        + rf"\bord{int(spec['highlight_outline_width'])}}}"
                        + segment
                        + "{"
                        + rf"\c{_ass_colour(text_bgr)}"
                        + rf"\3c{_ass_colour(outline_bgr)}"
                        + rf"\bord{outline_width}}}"
                    )
                else:
                    rendered_text_parts.append(segment)
        rendered_text = "".join(rendered_text_parts)
        start = _ass_time(int(decision.get("start_ms") or 0))
        end = _ass_time(int(decision.get("end_ms") or 0))
        background_shape = _ass_background_shape(
            text=text,
            font_size=font_size,
            position_x=position_x,
            center_y=center_y,
            alignment=str(spec["alignment"]),
            padding_x=int(spec["background_padding_x"]),
            padding_y=int(spec["background_padding_y"]),
            radius=int(spec["background_radius"]),
            background_bgr=spec["background_bgr"],
            background_opacity=float(
                spec.get("background_opacity", BOX_OPACITY)
            ),
            font_key=str(spec.get("font_key") or DEFAULT_FONT_KEY),
            frame_width=width,
            frame_height=height,
            minimum_width=int(spec.get("minimum_background_width") or 0),
            minimum_height=int(spec.get("minimum_background_height") or 0),
        )
        if bool(spec.get("draw_background", True)):
            lines.append(
                f"Dialogue: 0,{start},{end},VisualShape,,0,0,0,,{background_shape}"
            )
        lines.append(
            f"Dialogue: 1,{start},{end},VisualVI,,0,0,0,,{text_override}{rendered_text}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return target


def _escape_filter_path(path: Path) -> str:
    return (
        str(path.resolve())
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
    )


def _frame_intervals(
    decisions: list[dict[str, Any]],
    fps: float,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[str]]]:
    starts: dict[int, list[dict[str, Any]]] = {}
    ends: dict[int, list[str]] = {}
    for decision in decisions:
        exact_start = int(decision.get("start_frame") or 0)
        exact_end = int(decision.get("end_frame") or 0)
        if exact_start > 0 and exact_end >= exact_start:
            # Event profiles are 1-based and inclusive; renderer state is
            # 0-based and inclusive.
            start_frame = exact_start - 1
            end_frame = exact_end - 1
        else:
            start_frame = max(
                0,
                int(
                    math.floor(
                        float(decision.get("start_ms") or 0) * fps / 1000
                    )
                ),
            )
            # end_ms is exclusive.  Convert it to the final inclusive frame;
            # 0-40 ms at 25 fps therefore covers frame index 0, not 0 and 1.
            end_exclusive = max(
                start_frame + 1,
                int(
                    math.ceil(
                        float(decision.get("end_ms") or 0) * fps / 1000
                    )
                ),
            )
            end_frame = end_exclusive - 1
        starts.setdefault(start_frame, []).append(decision)
        ends.setdefault(end_frame + 1, []).append(str(decision["track_id"]))
    return starts, ends


class SinglePassLocalizedFrameRenderer:
    """Draw approved Vietnamese text before the STTN writer encodes a frame."""

    def __init__(
        self,
        review: dict[str, Any],
        *,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.review = review
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.specs = _direct_render_specs(
            review,
            width=self.width,
            height=self.height,
        )
        layout_preflight = _render_layout_issues(
            self.specs,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        if layout_preflight["overflow_track_ids"] or layout_preflight["overlap_pairs"]:
            raise SilentTextRenderError(
                "Localized layout preflight failed before STTN: "
                f"overflow={layout_preflight['overflow_track_ids']}; "
                f"overlap={layout_preflight['overlap_pairs']}"
            )
        self._layout_preflight = layout_preflight
        self._starts, self._ends = _frame_intervals(
            [spec["decision"] for spec in self.specs],
            self.fps,
        )
        prepare_started = time.perf_counter()
        self._patches_by_track = {}
        for spec in self.specs:
            patch = _prepare_direct_spec_patch(
                spec,
                frame_width=self.width,
                frame_height=self.height,
            )
            if patch is not None:
                self._patches_by_track[str(spec["track_id"])] = patch
        self._prepare_elapsed = time.perf_counter() - prepare_started
        self._active: dict[str, dict[str, Any]] = {}
        self._last_frame_index = -1
        self._submitted_frames = 0
        self._drawn_frames = 0
        self._draw_elapsed = 0.0

    def __call__(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if frame_index != self._last_frame_index + 1:
            raise SilentTextRenderError(
                "Single-pass renderer received frames out of order: "
                f"previous={self._last_frame_index}, current={frame_index}."
            )
        self._last_frame_index = frame_index
        self._submitted_frames += 1
        for track_id in self._ends.get(frame_index, []):
            self._active.pop(track_id, None)
        for decision in self._starts.get(frame_index, []):
            track_id = str(decision["track_id"])
            patch = self._patches_by_track.get(track_id)
            if patch is not None:
                self._active[track_id] = patch
        if not self._active:
            return frame
        started = time.perf_counter()
        rendered = _draw_prepared_direct_patches(
            frame,
            list(self._active.values()),
        )
        self._draw_elapsed += time.perf_counter() - started
        self._drawn_frames += 1
        return rendered

    def profile(self) -> dict[str, Any]:
        return {
            "single_pass": True,
            "submitted_frames": self._submitted_frames,
            "drawn_frames": self._drawn_frames,
            "draw_elapsed": self._draw_elapsed,
            "prepare_elapsed": self._prepare_elapsed,
            "render_anchor_count": len(self.specs),
            "layout_preflight": self._layout_preflight,
        }


def attach_single_pass_renderer_after_mask_selection(
    remover: Any,
    review: dict[str, Any],
) -> SinglePassLocalizedFrameRenderer | None:
    """Freeze render specs only after exact-mask guards finalize decisions."""
    if not getattr(remover, "single_pass_localized_render_requested", False):
        return None
    renderer = SinglePassLocalizedFrameRenderer(
        review,
        width=remover.frame_width,
        height=remover.frame_height,
        fps=remover.fps,
    )
    remover.output_frame_transform = renderer
    remover.single_pass_localized_renderer = renderer
    return renderer


def _encoder_args(encoder: str, preset: str, quality: int) -> list[str]:
    encoder = str(encoder or "auto").lower()
    preset = str(preset or "fast").lower()
    quality = max(0, min(51, int(quality)))
    # NVENC can corrupt static fine-text regions when a new direct-rendered
    # overlay appears elsewhere in a raw-frame stream.  Keep it available as
    # an explicit opt-in, but make "auto" use the stable x264 path.
    if encoder == "nvenc" and shutil.which("nvidia-smi"):
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "constqp",
            "-qp",
            str(quality),
        ]
    return [
        "-c:v",
        "libx264",
        "-preset",
        preset if preset in {"fast", "veryfast", "superfast", "ultrafast"} else "veryfast",
        "-crf",
        str(quality),
    ]


def _spec_geometry(spec: dict[str, Any]) -> dict[str, list[int]]:
    """Return the real Pillow text/background bounds before frame clipping."""
    font_size = int(spec["font_size"])
    font = _font(font_size, str(spec.get("font_key") or DEFAULT_FONT_KEY))
    metrics = []
    for line in str(spec["text"]).split(r"\N"):
        left, top, right, bottom = font.getbbox(line or "Ág")
        metrics.append(
            {
                "left": left,
                "top": top,
                "width": max(1, right - left),
                "height": max(1, bottom - top),
            }
        )
    line_gap = max(2, round(font_size * 0.12))
    text_height = sum(item["height"] for item in metrics) + line_gap * max(
        0, len(metrics) - 1
    )
    cursor_y = round(int(spec["center_y"]) - text_height / 2)
    placements = []
    for metric in metrics:
        if spec["alignment"] == "left":
            line_left = int(spec["position_x"])
        else:
            line_left = round(int(spec["position_x"]) - metric["width"] / 2)
        placements.append((metric, line_left, cursor_y))
        cursor_y += metric["height"] + line_gap
    outline = int(spec["outline_width"])
    text_box = [
        min(left for _metric, left, _top in placements) - outline,
        max(left + metric["width"] for metric, left, _top in placements)
        + outline,
        min(top for _metric, _left, top in placements) - outline,
        max(top + metric["height"] for metric, _left, top in placements)
        + outline,
    ]
    background_box = [
        text_box[0] - int(spec["background_padding_x"]),
        text_box[1] + int(spec["background_padding_x"]),
        text_box[2] - int(spec["background_padding_y"]),
        text_box[3] + int(spec["background_padding_y"]),
    ]
    background_box = _expand_background_box_to_minimum(
        background_box,
        minimum_width=int(spec.get("minimum_background_width") or 0),
        minimum_height=int(spec.get("minimum_background_height") or 0),
        center_x=int(spec["position_x"]),
        center_y=int(spec["center_y"]),
    )
    if spec.get("background_override_box") is not None:
        background_box = [
            int(value) for value in spec["background_override_box"]
        ]
    return {"text_box": text_box, "background_box": background_box}


def _boxes_intersect(first: list[int], second: list[int]) -> bool:
    return bool(
        min(first[1], second[1]) > max(first[0], second[0])
        and min(first[3], second[3]) > max(first[2], second[2])
    )


def _pycaps_safe_zone_box(
    *, frame_width: int, frame_height: int
) -> list[int]:
    """Return the band occupied by the locked production PyCaps profile."""
    return [
        0,
        int(frame_width),
        round(frame_height * PYCAPS_SAFE_ZONE_TOP_RATIO),
        round(frame_height * PYCAPS_SAFE_ZONE_BOTTOM_RATIO),
    ]


def _intrudes_pycaps_safe_zone(
    spec: dict[str, Any], *, frame_width: int, frame_height: int
) -> bool:
    if bool(spec.get("is_container")):
        return False
    return _boxes_intersect(
        _spec_geometry(spec)["background_box"],
        _pycaps_safe_zone_box(
            frame_width=frame_width,
            frame_height=frame_height,
        ),
    )


def _avoid_pycaps_safe_zone(
    specs: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Move generated editorial boxes away from the PyCaps dialogue band.

    Source cards are intentionally immovable because preserving their measured
    background and position is a stronger rule than the generated-box layout.
    """
    zone = _pycaps_safe_zone_box(
        frame_width=frame_width,
        frame_height=frame_height,
    )
    margin = max(4, round(frame_height * PYCAPS_SAFE_ZONE_MARGIN_RATIO))
    frame_margin = max(2, round(min(frame_width, frame_height) * 0.002))
    adjusted: list[dict[str, Any]] = []
    for original in specs:
        current = dict(original)
        if not _intrudes_pycaps_safe_zone(
            current,
            frame_width=frame_width,
            frame_height=frame_height,
        ):
            adjusted.append(current)
            continue
        box = _spec_geometry(current)["background_box"]
        shift_up = zone[2] - margin - box[3]
        shift_down = zone[3] + margin - box[2]
        candidates = sorted(
            (shift_up, shift_down),
            key=lambda value: abs(int(value)),
        )
        for delta_y in candidates:
            candidate = dict(current)
            candidate["center_y"] = int(current["center_y"]) + int(delta_y)
            candidate_box = _spec_geometry(candidate)["background_box"]
            if (
                candidate_box[2] < frame_margin
                or candidate_box[3] > frame_height - frame_margin
                or _boxes_intersect(candidate_box, zone)
            ):
                continue
            candidate["pycaps_safe_zone_adjusted"] = True
            candidate["pycaps_safe_zone_shift_y"] = int(delta_y)
            current = candidate
            break
        adjusted.append(current)
    return adjusted


def _fit_plain_specs_inside_frame(
    specs: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Shift movable fallback labels wholly inside the encoded frame."""
    fitted: list[dict[str, Any]] = []
    safe_margin = max(2, round(min(frame_width, frame_height) * 0.002))
    for original in specs:
        current = dict(original)
        if bool(current.get("is_container")):
            fitted.append(current)
            continue
        box = _spec_geometry(current)["text_box"]
        delta_x = 0
        delta_y = 0
        if box[0] < safe_margin:
            delta_x += safe_margin - box[0]
        if box[1] + delta_x > frame_width - safe_margin:
            delta_x += frame_width - safe_margin - (box[1] + delta_x)
        if box[2] < safe_margin:
            delta_y += safe_margin - box[2]
        if box[3] + delta_y > frame_height - safe_margin:
            delta_y += frame_height - safe_margin - (box[3] + delta_y)
        if delta_x or delta_y:
            current["position_x"] = int(current["position_x"]) + delta_x
            current["center_y"] = int(current["center_y"]) + delta_y
            current["frame_fit_adjusted"] = True
            current["frame_fit_shift_x"] = int(delta_x)
            current["frame_fit_shift_y"] = int(delta_y)
        fitted.append(current)
    return fitted


def _resolve_simultaneous_spec_overlaps(
    specs: list[dict[str, Any]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Move plain translated labels by the minimum amount needed to separate."""
    if len(specs) < 2:
        return specs
    resolved: list[tuple[int, dict[str, Any]]] = []
    margin = max(4, round(min(frame_width, frame_height) * 0.004))
    work = list(enumerate(specs))
    work.sort(
        key=lambda item: (
            not bool(item[1].get("is_container")),
            -int(item[1].get("font_size") or 0),
            item[0],
        )
    )
    for original_index, original in work:
        current = dict(original)
        blockers = [
            placed
            for _index, placed in resolved
            if _meaningful_render_overlap(
                current["decision"], placed["decision"]
            )
            and _boxes_intersect(
                _spec_geometry(current)["text_box"],
                _spec_geometry(placed)["text_box"],
            )
        ]
        if blockers and not bool(current.get("is_container")):
            original_x = int(current["position_x"])
            original_y = int(current["center_y"])
            current_box = _spec_geometry(current)["text_box"]
            candidates: list[tuple[int, int]] = []
            for blocker in blockers:
                blocker_box = _spec_geometry(blocker)["text_box"]
                candidates.extend(
                    [
                        (0, blocker_box[3] + margin - current_box[2]),
                        (0, blocker_box[2] - margin - current_box[3]),
                        (blocker_box[1] + margin - current_box[0], 0),
                        (blocker_box[0] - margin - current_box[1], 0),
                    ]
                )
            candidates.sort(key=lambda offset: abs(offset[0]) + abs(offset[1]))
            for delta_x, delta_y in candidates:
                candidate = dict(current)
                candidate["position_x"] = original_x + delta_x
                candidate["center_y"] = original_y + delta_y
                candidate_box = _spec_geometry(candidate)["text_box"]
                if (
                    candidate_box[0] < 0
                    or candidate_box[2] < 0
                    or candidate_box[1] > frame_width
                    or candidate_box[3] > frame_height
                ):
                    continue
                if _intrudes_pycaps_safe_zone(
                    candidate,
                    frame_width=frame_width,
                    frame_height=frame_height,
                ):
                    continue
                if any(
                    _meaningful_render_overlap(
                        candidate["decision"], placed["decision"]
                    )
                    and _boxes_intersect(
                        candidate_box,
                        _spec_geometry(placed)["text_box"],
                    )
                    for _index, placed in resolved
                ):
                    continue
                candidate["collision_adjusted"] = True
                candidate["collision_shift_x"] = int(delta_x)
                candidate["collision_shift_y"] = int(delta_y)
                current = candidate
                break
        resolved.append((original_index, current))
    return [spec for _index, spec in sorted(resolved, key=lambda item: item[0])]


def _render_layout_issues(
    specs: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fps: float,
) -> dict[str, Any]:
    geometry_by_track = {
        str(spec["track_id"]): _spec_geometry(spec) for spec in specs
    }
    overflow_track_ids = sorted(
        str(spec["track_id"])
        for spec in specs
        if (
            (box := geometry_by_track[str(spec["track_id"])]["text_box"])[0] < 0
            or box[2] < 0
            or box[1] > width
            or box[3] > height
        )
    )
    overlap_pairs: list[list[str]] = []
    for first, second in combinations(specs, 2):
        if not _meaningful_render_overlap(
            first["decision"], second["decision"], fps=fps
        ):
            continue
        first_id, second_id = str(first["track_id"]), str(second["track_id"])
        if _boxes_intersect(
            geometry_by_track[first_id]["text_box"],
            geometry_by_track[second_id]["text_box"],
        ):
            overlap_pairs.append([first_id, second_id])
    pycaps_safe_zone_intrusion_ids = sorted(
        str(spec["track_id"])
        for spec in specs
        if _intrudes_pycaps_safe_zone(
            spec,
            frame_width=width,
            frame_height=height,
        )
    )
    return {
        "passed": (
            not overflow_track_ids
            and not overlap_pairs
            and not pycaps_safe_zone_intrusion_ids
        ),
        "overflow_track_ids": overflow_track_ids,
        "overlap_pairs": overlap_pairs,
        "pycaps_safe_zone_intrusion_ids": pycaps_safe_zone_intrusion_ids,
    }


def _build_render_qc(
    review: dict[str, Any],
    specs: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    frame_count: int,
    fps: float,
) -> dict[str, Any]:
    decisions = list(review.get("decisions") or [])
    requested_replacements = [
        item
        for item in decisions
        if item.get("render_action") == "replace_with_vietnamese"
    ]
    missing_translation_ids = sorted(
        str(item.get("track_id") or "")
        for item in requested_replacements
        if not str(item.get("vietnamese_translation") or "").strip()
    )
    blocked_replacement_ids = sorted(
        str(item.get("track_id") or "")
        for item in requested_replacements
        if not _is_safe_vietnamese_replacement(item)
    )
    duplicate_track_ids = sorted(
        track_id
        for track_id in {str(item.get("track_id") or "") for item in specs}
        if sum(str(item.get("track_id") or "") == track_id for item in specs) > 1
    )
    minimum_readable_size = max(16, round(height * 0.012))
    too_small_ids = sorted(
        str(spec["track_id"])
        for spec in specs
        if int(spec["font_size"]) < minimum_readable_size
    )
    minimum_contrast_by_track: dict[str, float] = {}
    low_contrast_ids: list[str] = []
    specs_by_track = {str(spec["track_id"]): spec for spec in specs}
    for spec in specs:
        track_id = str(spec["track_id"])
        shared_owner = specs_by_track.get(
            str(spec.get("shared_background_owner") or "")
        )
        background_spec = (
            shared_owner if shared_owner and shared_owner.get("draw_background")
            else spec
        )
        if bool(background_spec.get("draw_background")):
            contrast = _translucent_background_contrast(
                spec["text_bgr"],
                background_spec["background_bgr"],
                float(background_spec.get("background_opacity", BOX_OPACITY)),
            )
        else:
            backgrounds = _container_background_samples(spec["decision"])
            contrast = min(
                (
                    _contrast_ratio(spec["text_bgr"], background)
                    for background in backgrounds
                ),
                default=0.0,
            )
        minimum_contrast_by_track[track_id] = round(float(contrast), 3)
        if contrast < MIN_TEXT_BACKGROUND_CONTRAST:
            low_contrast_ids.append(track_id)
    layout_issues = _render_layout_issues(
        specs,
        width=width,
        height=height,
        fps=fps,
    )
    overflow_ids = list(layout_issues["overflow_track_ids"])
    overlap_pairs = list(layout_issues["overlap_pairs"])
    pycaps_safe_zone_intrusion_ids = list(
        layout_issues["pycaps_safe_zone_intrusion_ids"]
    )
    duplicate_render_pairs = []
    for first, second in combinations(specs, 2):
        if not _meaningful_render_overlap(
            first["decision"],
            second["decision"],
            fps=fps,
        ):
            continue
        if _same_render_row_family(
            first,
            second,
            frame_width=width,
            frame_height=height,
        ):
            duplicate_render_pairs.append([first_id, second_id])
    coverage = review.get("mask_mapping_coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    unmapped_ids = sorted(
        str(item)
        for item in coverage.get("unmapped_selected_semantic_event_ids") or []
        if str(item)
    )
    preserved_unmapped_ids = sorted(
        str(item)
        for item in coverage.get("preserved_unmapped_semantic_event_ids") or []
        if str(item)
    )
    selected_count = sum(
        bool(item.get("selected_for_inpaint")) for item in decisions
    )
    selected_box_count = int(
        (review.get("mask_selection") or {}).get("selected_box_count") or 0
    )
    missing_mask_selection = bool(
        review.get("pre_inpaint_selection_applied")
        and selected_count
        and selected_box_count <= 0
    )
    errors = []
    if missing_translation_ids:
        errors.append("replacement_missing_vietnamese_translation")
    if blocked_replacement_ids:
        errors.append("unsafe_replacement_was_not_rendered")
    if duplicate_track_ids or duplicate_render_pairs:
        errors.append("duplicate_vietnamese_render")
    if overflow_ids:
        errors.append("rendered_text_overflows_frame")
    if overlap_pairs:
        errors.append("simultaneous_rendered_text_overlap")
    if pycaps_safe_zone_intrusion_ids:
        errors.append("rendered_text_intrudes_pycaps_safe_zone")
    if low_contrast_ids:
        errors.append("rendered_text_has_insufficient_contrast")
    if unmapped_ids or missing_mask_selection:
        errors.append("selected_event_has_no_exact_mask_coverage")
    warnings = []
    if too_small_ids:
        warnings.append("rendered_text_below_readability_advisory_size")
    if preserved_unmapped_ids:
        warnings.append("unmapped_event_preserved_before_sttn")
    collision_adjustments = [
        {
            "track_id": str(spec["track_id"]),
            "shift_x": int(spec.get("collision_shift_x") or 0),
            "shift_y": int(spec.get("collision_shift_y") or 0),
        }
        for spec in specs
        if bool(spec.get("collision_adjusted"))
    ]
    frame_fit_adjustments = [
        {
            "track_id": str(spec["track_id"]),
            "shift_x": int(spec.get("frame_fit_shift_x") or 0),
            "shift_y": int(spec.get("frame_fit_shift_y") or 0),
        }
        for spec in specs
        if bool(spec.get("frame_fit_adjusted"))
    ]
    pycaps_safe_zone_adjustments = [
        {
            "track_id": str(spec["track_id"]),
            "shift_y": int(spec.get("pycaps_safe_zone_shift_y") or 0),
        }
        for spec in specs
        if bool(spec.get("pycaps_safe_zone_adjusted"))
    ]
    return {
        "schema": "vsr.silent-text-render-qc",
        "version": 1,
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
        },
        "style_policy": {
            "preset": RENDER_STYLE_PRESET,
            "font_family": "semantic_role_mapping",
            "font_families": sorted(
                {str(spec.get("font_family") or DEFAULT_FONT_FAMILY) for spec in specs}
            ),
            "design_roles": sorted(
                {str(spec.get("design_role") or "neutral") for spec in specs}
            ),
            "colour_source": "measured_container_background_then_contrast",
            "preset_controls": [
                "semantic_font",
                "layout",
                "semantic_fallback_background",
            ],
            "measured_box_colour_overridden_by_preset": False,
            "source_container_preserved": True,
            "pycaps_profile_changed": False,
            "pycaps_safe_zone": {
                "top_ratio": PYCAPS_SAFE_ZONE_TOP_RATIO,
                "bottom_ratio": PYCAPS_SAFE_ZONE_BOTTOM_RATIO,
                "source_containers_exempt": True,
            },
        },
        "checks": {
            "requested_replacement_count": len(requested_replacements),
            "semantic_render_anchor_count": len(specs),
            "missing_translation_ids": missing_translation_ids,
            "blocked_replacement_ids": blocked_replacement_ids,
            "duplicate_track_ids": duplicate_track_ids,
            "duplicate_render_pairs": duplicate_render_pairs,
            "overflow_track_ids": sorted(set(overflow_ids)),
            "overlap_pairs": overlap_pairs,
            "collision_adjustments": collision_adjustments,
            "frame_fit_adjustments": frame_fit_adjustments,
            "pycaps_safe_zone_adjustments": pycaps_safe_zone_adjustments,
            "pycaps_safe_zone_intrusion_ids": pycaps_safe_zone_intrusion_ids,
            "minimum_readable_font_size": minimum_readable_size,
            "too_small_track_ids": too_small_ids,
            "minimum_required_text_contrast": MIN_TEXT_BACKGROUND_CONTRAST,
            "minimum_text_contrast_by_track": minimum_contrast_by_track,
            "low_contrast_track_ids": sorted(set(low_contrast_ids)),
            "selected_semantic_event_count": selected_count,
            "selected_exact_mask_box_count": selected_box_count,
            "unmapped_selected_semantic_event_ids": unmapped_ids,
            "preserved_unmapped_semantic_event_ids": preserved_unmapped_ids,
            "exact_mask_mapping_contract_complete": bool(
                coverage.get("complete", not unmapped_ids)
            ),
            "remaining_mask_check_scope": (
                "exact_mapping_coverage_plus_final_contact_sheet_visual_review"
            ),
        },
    }


def _even_samples(values: list[int], count: int) -> list[int]:
    values = sorted(set(values))
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return values
    if count == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (count - 1))]
        for index in range(count)
    ]


def _qc_frame_indices(
    review: dict[str, Any], *, frame_count: int, fps: float
) -> list[int]:
    desired = min(FINAL_QC_FRAME_COUNT, frame_count)
    event_frames = []
    for item in review.get("decisions") or []:
        if not item.get("selected_for_inpaint"):
            continue
        start = int(item.get("start_frame") or 0)
        end = int(item.get("end_frame") or 0)
        if start > 0 and end >= start:
            event_frames.append(max(0, min(frame_count - 1, (start + end) // 2 - 1)))
        else:
            midpoint_ms = (
                int(item.get("start_ms") or 0) + int(item.get("end_ms") or 0)
            ) / 2
            event_frames.append(
                max(0, min(frame_count - 1, round(midpoint_ms * fps / 1000)))
            )
    chosen = _even_samples(event_frames, min(6, desired))
    uniform = [
        round(index * (frame_count - 1) / max(1, desired - 1))
        for index in range(desired)
    ]
    for frame_index in uniform:
        if frame_index not in chosen:
            chosen.append(frame_index)
        if len(chosen) >= desired:
            break
    if len(chosen) < desired:
        for frame_index in range(frame_count):
            if frame_index not in chosen:
                chosen.append(frame_index)
            if len(chosen) >= desired:
                break
    return sorted(chosen)


def _write_final_contact_sheet(
    video_path: Path,
    output_path: Path,
    *,
    review: dict[str, Any],
    frame_count: int,
    fps: float,
) -> list[int]:
    frame_indices = _qc_frame_indices(review, frame_count=frame_count, fps=fps)
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    try:
        if not cap.isOpened():
            raise SilentTextRenderError("Cannot decode localized video for QC.")
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise SilentTextRenderError(
                    f"Cannot read localized QC frame {frame_index}."
                )
            frames.append((frame_index, frame))
    finally:
        cap.release()
    columns = 2
    cell_width, cell_height = 420, 292
    rows = max(1, math.ceil(len(frames) / columns))
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), "#101010"
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(16)
    for index, (frame_index, frame) in enumerate(frames):
        row, column = divmod(index, columns)
        x, y = column * cell_width, row * cell_height
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((cell_width - 16, cell_height - 38), Image.Resampling.LANCZOS)
        paste_x = x + (cell_width - image.width) // 2
        paste_y = y + 30 + (cell_height - 34 - image.height) // 2
        sheet.paste(image, (paste_x, paste_y))
        draw.text(
            (x + 8, y + 5),
            f"frame {frame_index + 1}  |  {frame_index / fps:.2f}s",
            fill="white",
            font=label_font,
        )
        draw.rectangle(
            (x, y, x + cell_width - 1, y + cell_height - 1),
            outline="#555555",
            width=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    sheet.save(temporary, "JPEG", quality=90, optimize=True)
    os.replace(temporary, output_path)
    return frame_indices


def finalize_single_pass_render(
    *,
    review: dict[str, Any],
    output_video: str | Path,
    renderer: SinglePassLocalizedFrameRenderer,
) -> dict[str, Any]:
    """Write the normal ASS/QC artifacts without decoding and encoding again."""
    target = Path(output_video)
    if not target.is_file() or target.stat().st_size <= 0:
        raise SilentTextRenderError("Single-pass localized video is missing.")
    cap = cv2.VideoCapture(str(target))
    try:
        if not cap.isOpened():
            raise SilentTextRenderError("Cannot decode single-pass localized video.")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
    finally:
        cap.release()
    renderer_profile = renderer.profile()
    if width != renderer.width or height != renderer.height:
        raise SilentTextRenderError(
            "Single-pass localized size mismatch: "
            f"encoded={(width, height)}, expected={(renderer.width, renderer.height)}."
        )
    if fps <= 0 or abs(fps - renderer.fps) > max(0.01, renderer.fps * 0.001):
        raise SilentTextRenderError(
            "Single-pass localized fps mismatch: "
            f"encoded={fps:.6f}, expected={renderer.fps:.6f}."
        )
    submitted_frames = int(renderer_profile["submitted_frames"])
    if frame_count != submitted_frames:
        raise SilentTextRenderError(
            "Single-pass localized frame count mismatch: "
            f"encoded={frame_count}, submitted={submitted_frames}."
        )
    ass = write_ass(
        review,
        target.parent / f"{target.stem}.silent-text.ass",
        width=width,
        height=height,
    )
    qc_path = target.with_suffix(".silent-text-qc.json")
    qc_contact_sheet = target.with_suffix(".silent-text-qc.jpg")
    qc_started = time.perf_counter()
    qc = _build_render_qc(
        review,
        renderer.specs,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
    )
    qc_frame_indices = _write_final_contact_sheet(
        target,
        qc_contact_sheet,
        review=review,
        frame_count=frame_count,
        fps=fps,
    )
    qc_elapsed = time.perf_counter() - qc_started
    qc["artifacts"] = {
        "contact_sheet": qc_contact_sheet.name,
        "report": qc_path.name,
        "contact_sheet_frame_count": len(qc_frame_indices),
        "contact_sheet_frame_indices": qc_frame_indices,
    }
    qc_path.write_text(
        json.dumps(qc, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not qc["passed"]:
        raise SilentTextRenderError(
            "Localized render QC failed: " + ", ".join(qc["errors"])
        )
    return {
        "output": str(target),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "ass": str(ass) if ass else None,
        "font": str(_font_path()),
        "render_backend": "pillow-direct-single-pass",
        "render_style_preset": RENDER_STYLE_PRESET,
        "qc_report": str(qc_path),
        "qc_contact_sheet": str(qc_contact_sheet),
        "qc": qc,
        "summary": review.get("summary") or {},
        "performance": {
            **renderer_profile,
            "source_frames_decoded": 0,
            "clean_frames_decoded_again": 0,
            "second_video_encode": False,
            "qc_elapsed": qc_elapsed,
        },
    }


def render_localized_video(
    *,
    review: dict[str, Any],
    source_video: str | Path,
    clean_video: str | Path,
    output_video: str | Path,
    ffmpeg: str = "ffmpeg",
    encoder: str = "auto",
    preset: str = "fast",
    quality: int = 20,
) -> dict[str, Any]:
    render_started = time.perf_counter()
    source = Path(source_video)
    clean = Path(clean_video)
    target = Path(output_video)
    if not source.is_file() or not clean.is_file():
        raise SilentTextRenderError("Source or clean video is missing.")
    source_cap = cv2.VideoCapture(str(source))
    clean_cap = cv2.VideoCapture(str(clean))
    try:
        if not source_cap.isOpened() or not clean_cap.isOpened():
            raise SilentTextRenderError("Cannot decode source or clean video.")
        width = int(clean_cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(clean_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        fps = float(clean_cap.get(cv2.CAP_PROP_FPS) or 0)
        clean_frame_count = int(
            clean_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5
        )
        if width <= 0 or height <= 0 or fps <= 0:
            raise SilentTextRenderError("Clean video has invalid metadata.")
        source_size = (
            int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
        )
        if source_size != (width, height):
            raise SilentTextRenderError(
                f"Source/clean size mismatch: {source_size} vs {(width, height)}."
            )
        source_fps = float(source_cap.get(cv2.CAP_PROP_FPS) or 0)
        source_frame_count = int(
            source_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5
        )
        if source_frame_count <= 0 or clean_frame_count <= 0:
            raise SilentTextRenderError("Source/clean frame count is invalid.")
        if source_frame_count != clean_frame_count:
            raise SilentTextRenderError(
                "Source/clean frame count mismatch: "
                f"{source_frame_count} vs {clean_frame_count}."
            )
        if source_fps <= 0 or abs(source_fps - fps) > max(0.01, source_fps * 0.001):
            raise SilentTextRenderError(
                f"Source/clean fps mismatch: {source_fps:.6f} vs {fps:.6f}."
            )
        ass = write_ass(
            review,
            target.parent / f"{target.stem}.silent-text.ass",
            width=width,
            height=height,
        )
        direct_specs = _direct_render_specs(
            review,
            width=width,
            height=height,
        )
        specs_by_track = {
            str(spec["track_id"]): spec for spec in direct_specs
        }
        replacement_starts, replacement_ends = _frame_intervals(
            [spec["decision"] for spec in direct_specs],
            fps,
        )
        restore_or_remove = [
            item
            for item in review.get("decisions") or []
            if item.get("render_action") in {
                "restore_original",
                "remove_original",
            }
            and not (
                review.get("pre_inpaint_selection_applied")
                and item.get("render_action") == "restore_original"
            )
        ]
        needs_source_frames = any(
            item.get("render_action") == "restore_original"
            for item in restore_or_remove
        )
        starts, ends = _frame_intervals(restore_or_remove, fps)
        active: dict[str, dict[str, Any]] = {}
        active_replacements: dict[str, dict[str, Any]] = {}
        video_temp = target.with_name(
            f".{target.stem}.localized-video{target.suffix}"
        )
        temp = target.with_name(f".{target.stem}.localized{target.suffix}")
        for stale_temp in (video_temp, temp):
            if stale_temp.exists():
                stale_temp.unlink()
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
        ]
        command.extend(
            [
                "-map",
                "0:v:0",
                *_encoder_args(encoder, preset, quality),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-frames:v",
                str(clean_frame_count),
                str(video_temp),
            ]
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stdin is None:
            raise SilentTextRenderError("Cannot open FFmpeg raw-video input.")
        frame_index = 0
        source_frames_decoded = 0
        try:
            while True:
                clean_ok, clean_frame = clean_cap.read()
                if not clean_ok:
                    break
                source_frame = None
                if needs_source_frames:
                    source_ok, source_frame = source_cap.read()
                    if not source_ok:
                        raise SilentTextRenderError(
                            "Source video ended before clean video at frame "
                            f"{frame_index}."
                        )
                    source_frames_decoded += 1
                for track_id in ends.get(frame_index, []):
                    active.pop(track_id, None)
                for decision in starts.get(frame_index, []):
                    active[str(decision["track_id"])] = decision
                for track_id in replacement_ends.get(frame_index, []):
                    active_replacements.pop(track_id, None)
                for decision in replacement_starts.get(frame_index, []):
                    track_id = str(decision["track_id"])
                    active_replacements[track_id] = specs_by_track[track_id]
                frame = clean_frame.copy()
                for render_action in ("restore_original", "remove_original"):
                    source_for_action = (
                        source_frame
                        if render_action == "restore_original"
                        else clean_frame
                    )
                    for decision in active.values():
                        if decision.get("render_action") != render_action:
                            continue
                        (x1, x2, y1, y2), _container = _event_box(
                            decision,
                            frame_width=width,
                            frame_height=height,
                            expand_plain=False,
                        )
                        frame[y1:y2, x1:x2] = source_for_action[
                            y1:y2,
                            x1:x2,
                        ]
                frame = _draw_direct_specs(
                    frame,
                    list(active_replacements.values()),
                )
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError as exc:
                    raise SilentTextRenderError(
                        "FFmpeg stopped while receiving localized frames."
                    ) from exc
                frame_index += 1
        finally:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if (
            return_code != 0
            or not video_temp.is_file()
            or video_temp.stat().st_size <= 0
        ):
            if video_temp.exists():
                video_temp.unlink()
            raise SilentTextRenderError(
                f"FFmpeg localized render failed ({return_code}): {stderr[-4000:]}"
            )
        if frame_index != clean_frame_count:
            if video_temp.exists():
                video_temp.unlink()
            raise SilentTextRenderError(
                "Localized render frame count mismatch: "
                f"decoded={frame_index}, expected={clean_frame_count}."
            )
        encoded_cap = cv2.VideoCapture(str(video_temp))
        try:
            encoded_frame_count = int(
                encoded_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5
            )
        finally:
            encoded_cap.release()
        if encoded_frame_count != clean_frame_count:
            if video_temp.exists():
                video_temp.unlink()
            raise SilentTextRenderError(
                "Localized encoded frame count mismatch: "
                f"encoded={encoded_frame_count}, expected={clean_frame_count}."
            )
        mux_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_temp),
            "-i",
            str(clean),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(temp),
        ]
        mux_result = subprocess.run(
            mux_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        video_temp.unlink(missing_ok=True)
        if (
            mux_result.returncode != 0
            or not temp.is_file()
            or temp.stat().st_size <= 0
        ):
            if temp.exists():
                temp.unlink()
            mux_stderr = mux_result.stderr.decode("utf-8", errors="replace")
            raise SilentTextRenderError(
                "FFmpeg localized audio mux failed "
                f"({mux_result.returncode}): {mux_stderr[-4000:]}"
            )
        muxed_cap = cv2.VideoCapture(str(temp))
        try:
            muxed_frame_count = int(
                muxed_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5
            )
        finally:
            muxed_cap.release()
        if muxed_frame_count != clean_frame_count:
            temp.unlink(missing_ok=True)
            raise SilentTextRenderError(
                "Localized muxed frame count mismatch: "
                f"muxed={muxed_frame_count}, expected={clean_frame_count}."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, target)
        qc_path = target.with_suffix(".silent-text-qc.json")
        qc_contact_sheet = target.with_suffix(".silent-text-qc.jpg")
        qc = _build_render_qc(
            review,
            direct_specs,
            width=width,
            height=height,
            frame_count=frame_index,
            fps=fps,
        )
        qc_frame_indices = _write_final_contact_sheet(
            target,
            qc_contact_sheet,
            review=review,
            frame_count=frame_index,
            fps=fps,
        )
        qc["artifacts"] = {
            "contact_sheet": qc_contact_sheet.name,
            "report": qc_path.name,
            "contact_sheet_frame_count": len(qc_frame_indices),
            "contact_sheet_frame_indices": qc_frame_indices,
        }
        qc_path.write_text(
            json.dumps(qc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not qc["passed"]:
            raise SilentTextRenderError(
                "Localized render QC failed: " + ", ".join(qc["errors"])
            )
        return {
            "output": str(target),
            "frame_count": frame_index,
            "width": width,
            "height": height,
            "fps": fps,
            "ass": str(ass) if ass else None,
            "font": str(_font_path()),
            "render_backend": "pillow-direct",
            "render_style_preset": RENDER_STYLE_PRESET,
            "qc_report": str(qc_path),
            "qc_contact_sheet": str(qc_contact_sheet),
            "qc": qc,
            "summary": review.get("summary") or {},
            "performance": {
                "single_pass": False,
                "source_frames_decoded": source_frames_decoded,
                "clean_frames_decoded_again": frame_index,
                "second_video_encode": True,
                "total_elapsed": time.perf_counter() - render_started,
            },
        }
    finally:
        source_cap.release()
        clean_cap.release()
