"""Gemini contract for translating visual text that is not spoken audio.

The network call is intentionally separated from validation. Kaggle tests can
exercise every safety rule with recorded/fake responses before a live key is
ever used.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend.tools.subtitle_event_profile import (
    MIN_AUTO_TRANSLATION_RENDER_SECONDS,
)
from backend.tools.subtitle_speech_alignment import (
    LOCAL_SPOKEN_ROUTE_MAX_DURATION_RATIO,
    LOCAL_SPOKEN_ROUTE_MIN_SCORE,
    LOCAL_SPOKEN_ROUTE_MIN_TEXT_LENGTH_RATIO,
    align_subtitle_events_with_srt,
    annotate_speech_review_routes,
)
from backend.tools.visual_text_style import DESIGN_ROLES, infer_design_role

REVIEW_SCHEMA = "vsr.gemini-silent-text-review"
REVIEW_VERSION = 1
SILENT_HARD_SUB_CLASSES = {
    "silent_dialogue",
    "narrative_text",
}
# Backwards-compatible public name used by existing callers. Translation is
# intentionally narrower than the complete set of non-spoken text classes.
TRANSLATABLE_CLASSES = SILENT_HARD_SUB_CLASSES
TEXT_ORIGINS = {
    "post_production_overlay",
    "scene_text",
    "ui_or_watermark",
    "uncertain",
}
CLASSIFICATIONS = {
    "spoken_subtitle",
    "silent_dialogue",
    "narrative_text",
    "sign_or_ui",
    "decorative_overlay",
    "uncertain",
}
REPLACEMENT_ACTIONS = {
    "replace_with_vietnamese",
    "keep_removed",
    "ignore",
    "needs_review",
}
CONTACT_SHEET_EVENTS = 8
GEMINI_REVIEW_BATCH_EVENTS = 32
SCENERY_GUARD_MAX_CANDIDATES = 64
CONTACT_SHEET_SIZE = (1520, 1720)
CONTACT_CELL_SIZE = (760, 430)
GEMINI_API_ROOT = "https://generativelanguage.googleapis.com"
API_KEY_BUNDLE_SCHEMA = "vsr.gemini-api-keys"
API_KEY_BUNDLE_VERSION = 1
PRIMARY_GEMINI_MODEL = "gemini-3.6-flash"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash"
FALLBACK_GEMINI_MODELS = (FALLBACK_GEMINI_MODEL,)
SCENERY_GUARD_MODEL = "gemini-3.5-flash-lite"
GEMINI_REVIEW_MODELS = (
    PRIMARY_GEMINI_MODEL,
    *FALLBACK_GEMINI_MODELS,
)
SCENERY_NOT_TEXT_CONFIDENCE = 0.97
FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE = 0.72
FOCUSED_REVIEW_OCR_SUPPORT_MIN_TEXT_LENGTH = 2
PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_DURATION_SECONDS = 0.16
PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_OBSERVED_FRAMES = 4
SEMANTIC_FRAGMENTATION_GUARD_MIN_BATCHES = 9
SEMANTIC_FRAGMENTATION_GUARD_MAX_EVENTS_PER_SECOND = 4.0
SEMANTIC_FRAGMENTATION_GUARD_MIN_RETENTION_RATIO = 0.78
LITE_TRIAGE_MODEL = "gemini-3.5-flash-lite"
ANTIGRAVITY_TRIAGE_PROVIDER = "antigravity"
ANTIGRAVITY_AGENT = "antigravity-preview-05-2026"
ANTIGRAVITY_TRIAGE_MODEL = "gemini-3.6-flash"
ANTIGRAVITY_TRIAGE_TOKEN_BUDGET = 3200
INLINE_REVIEW_TIMEOUT_SECONDS = 60
MODEL_503_RETRY_LIMIT = 3
MAX_503_KEY_CYCLES = 10
MODEL_503_BACKOFF_SECONDS = (1.0, 2.0)
UNKNOWN_429_COOLDOWN_SECONDS = 15.0
QUOTA_429_COOLDOWN_SECONDS = 60.0
GENERIC_TRANSIENT_RETRY_LIMIT = 3
MIDDLE_MICROTEXT_MAX_DURATION_SECONDS = 0.8
MIDDLE_MICROTEXT_MAX_AREA_FRACTION = 0.012
MIDDLE_MICROTEXT_MAX_HEIGHT_FRACTION = 0.055
MIDDLE_MICROTEXT_TOP_FRACTION = 0.22
MIDDLE_MICROTEXT_BOTTOM_FRACTION = 0.78
# One Gemini semantic decision may cover repeated appearances of the same label,
# but rendering must not bridge a real disappearance. Keep this aligned with
# the duplicate-render continuity guard: gaps above 420 ms are separate visible
# occurrences while still sharing the same semantic decision and translation.
SEMANTIC_RENDER_COMPONENT_MAX_GAP_MS = 420
# Persistent edge overlays can disappear from OCR for one or two scene cuts.
# Keep the bridge below three seconds so unrelated edge text is not casually
# merged, while preserving the 62 s acceptance clip's ~2.2 s detector gaps.
PERSISTENT_VISUAL_MAX_GAP_MS = 2500
# Exact-mask evidence is used to extend a reviewed persistent overlay timeline.
# Keep this stricter than visual clustering so a detached earlier edge object
# cannot be bridged into the canonical removal span.
PERSISTENT_EXACT_EVIDENCE_MAX_GAP_MS = 2000


class SilentTextLocalizationError(ValueError):
    """Raised when Gemini output or localization configuration is unsafe."""


class GeminiRequestError(SilentTextLocalizationError):
    """A safe Gemini transport error that never contains an API key."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        rotate_key: bool = False,
        quota_dimension: str = "",
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.rotate_key = rotate_key
        self.quota_dimension = str(quota_dimension or "").upper()
        self.retry_after_seconds = retry_after_seconds


class GeminiPolicyBlockError(SilentTextLocalizationError):
    """Gemini accepted the request but policy-blocked its content."""

    def __init__(self, block_reason: str) -> None:
        self.block_reason = str(block_reason or "UNKNOWN").strip().upper()
        super().__init__(f"Gemini policy blocked content ({self.block_reason}).")


def normalize_localization_config(value: object) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    enabled = bool(data.get("enabled", False))
    auto_approve = enabled and bool(data.get("auto_approve", enabled))
    auto_render = enabled and bool(data.get("auto_render", auto_approve))
    scenery_guard_enabled = enabled and bool(
        data.get("scenery_guard_enabled", False)
    )
    focused_second_review_enabled = enabled and bool(
        data.get("focused_second_review_enabled", True)
    )
    middle_microtext_filter_enabled = enabled and bool(
        data.get("middle_microtext_filter_enabled", False)
    )
    lite_parallel_triage_enabled = enabled and bool(
        data.get("lite_parallel_triage_enabled", False)
    )
    parallel_triage_provider = str(
        data.get("parallel_triage_provider") or "gemini"
    ).strip().lower()
    if parallel_triage_provider not in {"gemini", ANTIGRAVITY_TRIAGE_PROVIDER}:
        raise SilentTextLocalizationError(
            "parallel_triage_provider must be gemini or antigravity."
        )
    parallel_triage_model = str(
        data.get("parallel_triage_model") or FALLBACK_GEMINI_MODEL
    ).strip()
    if (
        parallel_triage_provider == "gemini"
        and parallel_triage_model not in {LITE_TRIAGE_MODEL, FALLBACK_GEMINI_MODEL}
    ):
        raise SilentTextLocalizationError(
            "parallel_triage_model must be gemini-3.5-flash-lite or gemini-3.5-flash."
        )
    try:
        threshold = float(data.get("confidence_threshold", 0.90))
    except (TypeError, ValueError) as exc:
        raise SilentTextLocalizationError(
            "confidence_threshold must be a number."
        ) from exc
    if not 0.50 <= threshold <= 1.0:
        raise SilentTextLocalizationError(
            "confidence_threshold must be between 0.50 and 1.00."
        )
    def _bounded_number(name: str, default: float, low: float, high: float) -> float:
        try:
            number = float(data.get(name, default))
        except (TypeError, ValueError) as exc:
            raise SilentTextLocalizationError(f"{name} must be a number.") from exc
        if not low <= number <= high:
            raise SilentTextLocalizationError(
                f"{name} must be between {low:g} and {high:g}."
            )
        return number

    middle_microtext_max_duration_seconds = _bounded_number(
        "middle_microtext_max_duration_seconds",
        MIDDLE_MICROTEXT_MAX_DURATION_SECONDS,
        0.04,
        10.0,
    )
    middle_microtext_max_area_fraction = _bounded_number(
        "middle_microtext_max_area_fraction",
        MIDDLE_MICROTEXT_MAX_AREA_FRACTION,
        0.0001,
        0.25,
    )
    middle_microtext_max_height_fraction = _bounded_number(
        "middle_microtext_max_height_fraction",
        MIDDLE_MICROTEXT_MAX_HEIGHT_FRACTION,
        0.0001,
        0.5,
    )
    middle_microtext_top_fraction = _bounded_number(
        "middle_microtext_top_fraction",
        MIDDLE_MICROTEXT_TOP_FRACTION,
        0.0,
        0.95,
    )
    middle_microtext_bottom_fraction = _bounded_number(
        "middle_microtext_bottom_fraction",
        MIDDLE_MICROTEXT_BOTTOM_FRACTION,
        0.05,
        1.0,
    )
    if middle_microtext_top_fraction >= middle_microtext_bottom_fraction:
        raise SilentTextLocalizationError(
            "middle_microtext_top_fraction must be below middle_microtext_bottom_fraction."
        )
    lite_parallel_batch_size = int(
        _bounded_number("lite_parallel_batch_size", 8, 1, 32)
    )
    lite_parallel_workers = int(
        _bounded_number("lite_parallel_workers", 10, 1, 16)
    )
    strong_parallel_workers = int(
        _bounded_number("strong_parallel_workers", 4, 1, 4)
    )
    lite_safe_keep_confidence = _bounded_number(
        "lite_safe_keep_confidence", 0.90, 0.90, 1.0
    )
    first_secret = str(
        data.get("api_key_secret") or "GEMINI_API_KEY"
    ).strip()
    raw_secrets = data.get("api_key_secrets")
    if raw_secrets is None:
        raw_secrets = [
            first_secret,
            f"{first_secret}_2",
            f"{first_secret}_3",
        ]
    if not isinstance(raw_secrets, (list, tuple)):
        raise SilentTextLocalizationError(
            "api_key_secrets must be a list."
        )
    secret_names: list[str] = []
    for item in raw_secrets:
        name = str(item or "").strip()
        if (
            not re.fullmatch(r"[A-Z][A-Z0-9_]{2,63}", name)
            or name in secret_names
        ):
            raise SilentTextLocalizationError(
                "api_key_secrets contains an invalid or duplicate name."
            )
        secret_names.append(name)
    if not 1 <= len(secret_names) <= 10:
        raise SilentTextLocalizationError(
            "api_key_secrets supports 1-10 names."
        )
    key_file_path = str(data.get("api_key_file_path") or "").strip()
    key_file_sha256 = str(data.get("api_key_file_sha256") or "").strip().lower()
    if key_file_sha256 and not re.fullmatch(r"[0-9a-f]{64}", key_file_sha256):
        raise SilentTextLocalizationError("api_key_file_sha256 is invalid.")
    return {
        "enabled": enabled,
        "auto_approve": auto_approve,
        "auto_render": auto_render,
        "confidence_threshold": round(threshold, 4),
        "target_language": "vi",
        "gemini_model": PRIMARY_GEMINI_MODEL,
        "gemini_fallback_model": FALLBACK_GEMINI_MODEL,
        "gemini_fallback_models": list(FALLBACK_GEMINI_MODELS),
        "scenery_guard_model": SCENERY_GUARD_MODEL,
        "scenery_guard_enabled": scenery_guard_enabled,
        "focused_second_review_enabled": focused_second_review_enabled,
        "middle_microtext_filter_enabled": middle_microtext_filter_enabled,
        "lite_parallel_triage_enabled": lite_parallel_triage_enabled,
        "parallel_triage_provider": parallel_triage_provider,
        "parallel_triage_model": parallel_triage_model,
        "lite_parallel_batch_size": lite_parallel_batch_size,
        "lite_parallel_workers": lite_parallel_workers,
        "strong_parallel_workers": strong_parallel_workers,
        "lite_safe_keep_confidence": round(lite_safe_keep_confidence, 4),
        "middle_microtext_max_duration_seconds": round(
            middle_microtext_max_duration_seconds, 4
        ),
        "middle_microtext_max_area_fraction": round(
            middle_microtext_max_area_fraction, 6
        ),
        "middle_microtext_max_height_fraction": round(
            middle_microtext_max_height_fraction, 6
        ),
        "middle_microtext_top_fraction": round(middle_microtext_top_fraction, 4),
        "middle_microtext_bottom_fraction": round(
            middle_microtext_bottom_fraction, 4
        ),
        "api_key_secret": secret_names[0],
        "api_key_secrets": secret_names,
        "api_key_file_path": key_file_path,
        "api_key_file_sha256": key_file_sha256,
    }


def load_subtitle_event_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        profile = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SilentTextLocalizationError(
            f"Cannot read subtitle event profile: {source}"
        ) from exc
    events = profile.get("events")
    if (
        profile.get("schema") != "vsr.subtitle-events"
        or profile.get("version") != 1
        or not isinstance(events, list)
        or profile.get("event_count") != len(events)
    ):
        raise SilentTextLocalizationError("Invalid VSR subtitle event profile.")
    track_ids = [event.get("track_id") for event in events]
    if (
        any(not isinstance(track_id, str) or not track_id for track_id in track_ids)
        or len(track_ids) != len(set(track_ids))
    ):
        raise SilentTextLocalizationError(
            "VSR subtitle events contain invalid or duplicate track_id values."
        )
    return profile


def gemini_response_schema() -> dict[str, Any]:
    text = {"type": "string"}
    item_properties = {
        "track_id": text,
        "classification": {
            "type": "string",
            "enum": sorted(CLASSIFICATIONS),
        },
        "text_origin": {
            "type": "string",
            "enum": sorted(TEXT_ORIGINS),
        },
        "source_text_corrected": text,
        "vietnamese_translation": text,
        "highlight_terms": {
            "type": "array",
            "items": text,
            "maxItems": 2,
        },
        "design_role": {
            "type": "string",
            "enum": sorted(DESIGN_ROLES),
        },
        "replacement_action": {
            "type": "string",
            "enum": sorted(REPLACEMENT_ACTIONS),
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reason_vi": text,
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                },
            }
        },
        "required": ["items"],
    }


def scenery_guard_response_schema() -> dict[str, Any]:
    """Schema for the narrow OCR false-positive guard.

    Flash Lite is intentionally not asked to read, translate, classify or
    approve text.  It only decides whether the marked shape is made of actual
    character glyphs or is scenery that OCR mistook for glyphs.
    """
    item = {
        "type": "object",
        "properties": {
            "track_id": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["NOT_TEXT", "REAL_TEXT", "UNCERTAIN"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_vi": {"type": "string"},
        },
        "required": ["track_id", "verdict", "confidence", "reason_vi"],
    }
    return {
        "type": "object",
        "properties": {"items": {"type": "array", "items": item}},
        "required": ["items"],
    }


def _scenery_guard_candidates(profile: dict[str, Any]) -> list[dict[str, Any]]:
    video = profile.get("video") or {}
    width = max(1, int(video.get("width") or 1))
    height = max(1, int(video.get("height") or 1))
    candidates = []
    for event in profile.get("events") or []:
        # Old downloaded bundles do not have exact mask provenance.  They stay
        # on the compatible 3.6-only path instead of being guessed about.
        if not isinstance(event.get("frame_boxes"), list):
            continue
        box = event.get("bbox") or []
        if len(box) != 4:
            continue
        xmin, xmax, ymin, ymax = (int(value) for value in box)
        box_width = max(1, xmax - xmin)
        box_height = max(1, ymax - ymin)
        center_x = (xmin + xmax) / (2 * width)
        center_y = (ymin + ymax) / (2 * height)
        normalized_text = "".join(
            character
            for character in str(event.get("ocr_best_text") or "")
            if character.isalnum()
        )
        confidence = float(event.get("ocr_best_confidence") or 0.0)
        observed = int(event.get("observed_frames") or 0)
        sources = set(event.get("detection_sources") or [])
        subtitle_shaped = bool(
            0.16 <= center_x <= 0.84
            and center_y >= 0.44
            and box_width / width >= 0.07
            and box_width / box_height >= 1.15
            and observed >= 2
            and confidence >= 0.72
            and len(normalized_text) >= 2
        )
        weak_or_sparse = bool(
            confidence < 0.78
            or len(normalized_text) < 2
            or observed <= 1
            or sources == {"stage7"}
        )
        if weak_or_sparse or not subtitle_shaped:
            candidates.append(event)
    return candidates


def build_scenery_guard_prompt(events: list[dict[str, Any]]) -> str:
    identifiers = [str(event.get("track_id") or "") for event in events]
    return (
        "Bạn là bộ lọc chống OCR nhận nhầm cảnh vật. Mỗi ảnh đính kèm có một "
        "khung vàng và ô phóng to tương ứng với track_id theo đúng thứ tự sau: "
        f"{json.dumps(identifiers, ensure_ascii=False)}. "
        "Chỉ xác định hình dạng trong khung có thực sự gồm các ký tự/chữ số được "
        "con người viết hoặc đồ họa tạo ra hay chỉ là cành cây, lá, cỏ, vân đất, "
        "mặt nước, bóng, vật thể hoặc nhiễu hình ảnh giống nét chữ. KHÔNG đọc nội "
        "dung, KHÔNG dịch, KHÔNG phân loại loại chữ, KHÔNG quyết định xóa/render. "
        "Dùng NOT_TEXT chỉ khi chắc chắn đó là cảnh vật/nhiễu; REAL_TEXT khi chắc "
        "chắn có glyph thật; còn lại dùng UNCERTAIN. Trả đúng một item cho mỗi "
        "track_id và chỉ trả JSON theo schema. reason_vi giải thích rất ngắn."
    )


def validate_scenery_guard_response(
    events: list[dict[str, Any]],
    response: str | dict[str, Any],
) -> dict[str, dict[str, Any]]:
    payload = _json_object(response)
    items = payload.get("items")
    if not isinstance(items, list):
        raise SilentTextLocalizationError("Scenery guard response is missing items.")
    expected = {str(event.get("track_id") or "") for event in events}
    results: dict[str, dict[str, Any]] = {}
    for raw in items:
        if not isinstance(raw, dict):
            raise SilentTextLocalizationError("Scenery guard item is invalid.")
        track_id = str(raw.get("track_id") or "")
        verdict = str(raw.get("verdict") or "")
        confidence = raw.get("confidence")
        reason = str(raw.get("reason_vi") or "").strip()
        if (
            track_id not in expected
            or track_id in results
            or verdict not in {"NOT_TEXT", "REAL_TEXT", "UNCERTAIN"}
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
            or not reason
        ):
            raise SilentTextLocalizationError(
                f"Scenery guard returned an invalid item for {track_id!r}."
            )
        results[track_id] = {
            "track_id": track_id,
            "verdict": verdict,
            "confidence": float(confidence),
            "reason_vi": reason,
        }
    if set(results) != expected:
        raise SilentTextLocalizationError("Scenery guard track_id set mismatch.")
    return results


def build_gemini_prompt(
    profile: dict[str, Any],
    *,
    srt_text: str = "",
) -> str:
    events = [
        {
            "track_id": event["track_id"],
            "start_ms": int(event.get("start_ms") or 0),
            "end_ms": int(event.get("end_ms") or 0),
            "bbox_normalized": event.get("bbox_normalized") or [],
            "ocr_best_text": str(event.get("ocr_best_text") or ""),
            "ocr_best_confidence": float(event.get("ocr_best_confidence") or 0),
            "ocr_candidates": event.get("ocr_candidates") or [],
            "speech_relation": event.get("speech_relation") or None,
            "scenery_guard_hint": event.get("scenery_guard_hint") or None,
        }
        for event in profile["events"]
    ]
    srt_section = srt_text.strip() or "(Không có SRT rời.)"
    return f"""
Bạn là biên tập viên phụ đề đa phương thức. Bằng chứng được cung cấp chỉ gồm
contact sheet VSR (ảnh toàn khung và crop đại diện), OCR, thời gian/toạ độ và
SRT nếu có. Không có tệp video liên tục và không có audio đính kèm. Không được
suy đoán rằng bạn đã xem/nghe dữ liệu không được cung cấp. Khi các ảnh đại diện,
OCR, speech_relation và SRT vẫn chưa đủ để phân biệt, hãy trả uncertain.

Trường hợp đặc biệt cần kiểm tra kỹ: nếu ba track persistent có gần như cùng
bbox, track ngắn ở đầu và track ngắn ở cuối có cùng OCR/nội dung, còn track giữa
chạy qua khoảng thời gian ở giữa, thì hai track ngắn thường là hai mảnh của dòng
trên và track giữa là dòng khác ở ngay bên dưới. Khi OCR của track giữa lặp lại
dòng trên, không được gán lại dòng trên cho track giữa. Hãy đọc dòng bên dưới từ
crop phóng lớn và dịch đúng dòng đó. Nếu crop vẫn không đủ rõ, trả uncertain cho
track giữa thay vì đoán hoặc nhân đôi bản dịch.

Mục tiêu: dịch mọi chữ đồ hoạ biên tập có ý nghĩa được chèn hậu kỳ lên video mà
không thể hiện lời đang nghe: phụ đề im lặng, chú thích, tiêu đề, nhãn series và
tên kênh/chủ đề được trình bày lớn trong một thẻ/hộp thiết kế. Không dịch chữ vốn thuộc cảnh quay,
chữ trên vật thể, biển hiệu, UI nền tảng, logo hình hoặc watermark
nhận diện nguồn nhỏ.

Nhãn:
- spoken_subtitle: phụ đề khớp lời thoại theo SRT/speech_relation được cung cấp.
- silent_dialogue: lời thoại/chữ nhân vật được chèn hậu kỳ nhưng không có câu nói
  tương ứng trong SRT/speech_relation được cung cấp.
- narrative_text: chữ dẫn chuyện/chú thích nội dung được chèn hậu kỳ lên video.
- sign_or_ui: chữ thuộc cảnh, biển hiệu, màn hình thiết bị, tin nhắn, nút hoặc UI.
- decorative_overlay: logo hình, watermark/tên tài khoản nhỏ, quảng cáo lặp hoặc
  chữ trang trí không mang thông tin. Tên kênh/chủ đề viết lớn trong thẻ đồ hoạ
  biên tập là narrative_text, không phải decorative_overlay.
- uncertain: bằng chứng chưa đủ chắc.

Nguồn gốc chữ (text_origin):
- post_production_overlay: lớp chữ đồ hoạ chèn hậu kỳ; thường bám màn hình, có
  kiểu phụ đề/chú thích và không chuyển động theo vật thể trong cảnh.
- scene_text: chữ tồn tại vật lý trong cảnh hoặc bám theo vật thể/không gian cảnh.
- ui_or_watermark: giao diện nền tảng, logo, watermark nhỏ, @tên_tài_khoản hoặc
  lớp nhận diện nguồn. Tên kênh/series trình bày lớn trong hộp đồ hoạ và truyền đạt
  chủ đề cho người xem là post_production_overlay.
- uncertain: không đủ bằng chứng để xác định nguồn gốc.

"Watermark" ở đây chỉ là lớp nhận diện nguồn/thương hiệu/tài khoản hoặc logo.
Một câu hoàn chỉnh như lời miễn trừ trách nhiệm, cảnh báo nội dung, ghi chú của
biên tập viên hoặc chú thích thông tin KHÔNG trở thành watermark chỉ vì nó nhỏ,
mờ, nằm ở mép khung hay tồn tại suốt video. Nếu câu đó là lớp chữ phẳng được
chèn lên video thì phải là narrative_text + post_production_overlay và được dịch.
Tương tự, dấu # không tự biến một nhãn hành động/cảm xúc ngắn thành watermark.
Các nhãn như "#đánh răng", "#không có bọt" mô tả việc đang xảy ra là chú thích
hậu kỳ và phải dịch. Chỉ giữ dạng # khi đó thực sự là @tên tài khoản, logo hoặc
nhận diện nguồn nhỏ lặp lại. Tên kênh/series lớn, dễ đọc, nằm trong thẻ thiết kế
phải được Việt hoá.

Phân biệt theo cách chữ hiển thị, KHÔNG theo nghĩa của câu. Dòng chữ cảnh báo,
hướng dẫn, luật lệ hoặc an toàn nếu là lớp chèn phẳng trên khung hình (không bám
phối cảnh/vật thể, không bị che khuất và không chuyển động theo camera) thì vẫn là
post_production_overlay + narrative_text và phải dùng replace_with_vietnamese. Ví dụ:
"đã thắt dây an toàn", "lái xe đoạn đường an toàn" chèn ở đáy video là chú thích
hậu kỳ, không phải biển báo trong cảnh. Chỉ dùng scene_text/sign_or_ui khi chữ thực sự
nằm trên biển/vật thể/màn hình trong cảnh hoặc bám theo phối cảnh.

Quy tắc bắt buộc:
- Khi nhiều track OCR cùng hộp nhưng ảnh crop cho thấy nhiều dòng chữ riêng,
  hãy gán mỗi track cho đúng dòng nhìn thấy theo hình học và thời gian; OCR có
  thể đọc lặp dòng đầu khi hộp bao hai dòng. Không được gán lặp dòng đầu cho
  track giữa rồi bỏ sót dòng kế tiếp. Nếu không xác định được dòng thuộc track
  nào từ ảnh, trả uncertain thay vì đoán chữ hoặc bản dịch.
- Trả đúng một item cho mỗi track_id; không thiếu, không thêm, không trùng.
- Không thay đổi thời gian hoặc toạ độ.
- spoken_subtitle phải dùng keep_removed và không có bản dịch để chèn.
- replace_with_vietnamese chỉ được dùng khi classification là silent_dialogue
  hoặc narrative_text VÀ text_origin là post_production_overlay.
- sign_or_ui, scene_text, ui_or_watermark và decorative_overlay phải dùng ignore;
  VSR sẽ giữ/khôi phục chữ gốc, không chèn tiếng Việt.
- Nếu chưa chắc chữ là lớp chèn hậu kỳ, dùng text_origin=uncertain,
  classification=uncertain và replacement_action=needs_review.
- replace_with_vietnamese bắt buộc có vietnamese_translation đầy đủ.
- vietnamese_translation là câu dùng để in trực tiếp lên video: phải tự nhiên,
  súc tích và không thêm lời giải thích hoặc từ đồng nghĩa dư thừa.
- highlight_terms chỉ dành cho replace_with_vietnamese: trả về 0-2 từ/cụm từ
  quan trọng xuất hiện nguyên văn trong vietnamese_translation. Không highlight
  cả câu và không vượt quá khoảng 40% nội dung. Nếu không có từ nhấn mạnh rõ
  ràng, trả về mảng rỗng [].
- Với tiêu đề, nhãn hoặc narrative overlay ngắn có một ý chính rõ ràng, ưu tiên
  chọn đúng 1 từ/cụm từ mang ý nghĩa nổi bật; không chọn các từ chức năng phổ
  thông chỉ để bắt buộc có highlight.
- design_role mô tả vai trò trình bày, không phải chủ đề video:
  headline = tiêu đề/hook lớn; explanatory = giải thích/thuyết minh dài;
  metric_badge = giá, phần trăm, số liệu hoặc ưu đãi; label_chip = nhãn ngắn;
  silent_dialogue = câu thoại im lặng; playful_emphasis = cảm xúc/hài hước;
  neutral = mọi item không được replace_with_vietnamese.
- Với replace_with_vietnamese phải chọn design_role phù hợp nhất. Với mọi action
  khác bắt buộc dùng neutral. Font và nền sẽ do renderer quyết định ổn định từ
  design_role; không mô tả màu/font trong reason_vi.
- Giữ nghĩa chính nhưng ưu tiên độ dài gần chữ nguồn. Với nhãn, tiêu đề hoặc
  chữ trong hộp, hãy chọn cách diễn đạt tiếng Việt ngắn nhất vẫn rõ nghĩa.
- uncertain phải dùng needs_review.
- decorative_overlay dùng ignore; không gán nhãn này cho chữ tên kênh/chủ đề lớn
  trong một thẻ/hộp đồ hoạ biên tập có nội dung người xem cần hiểu.
- Không được dùng sign_or_ui/scene_text chỉ vì nội dung là cảnh báo, hướng dẫn
  hoặc thông tin pháp lý. Chỉ giữ nguyên khi bằng chứng hình ảnh cho thấy chữ thuộc
  cảnh quay.
- Không dùng decorative_overlay/ui_or_watermark cho câu disclaimer, content
  warning, chú thích nội dung hoặc thông báo biên tập. decorative_overlay chỉ dành
  cho nhận diện nguồn/thương hiệu/quảng cáo lặp/chữ trang trí không mang nội dung
  tường thuật của video.
- Không bịa chữ khi ảnh/OCR không đủ rõ; không suy đoán nội dung audio.
- scenery_guard_hint từ Gemini Flash Lite chỉ là gợi ý phát hiện OCR nhầm cảnh
  vật, KHÔNG phải quyết định cuối cùng và có thể sai dù confidence rất cao. Luôn
  tự kiểm tra lại contact sheet. Nếu trong hoặc cắt qua khung còn bất kỳ glyph, nét
  chữ hay viền chữ nào thì phải xử lý đó là chữ thật; không được giữ cảnh chỉ vì
  Lite trả NOT_TEXT.
- confidence phản ánh độ chắc chắn của cả phân loại lẫn bản dịch.
- reason_vi luôn giải thích ngắn gọn bằng tiếng Việt.
- speech_relation là kết quả đối chiếu OCR-SRT cục bộ. Nó chỉ cho biết quan hệ
  với lời thoại, không tự chứng minh chữ thuộc cảnh hay là lớp chèn hậu kỳ.
- Chỉ trả JSON đúng schema, không thêm Markdown.

SRT lời thoại:
---BEGIN SRT---
{srt_section}
---END SRT---

Sự kiện VSR:
{json.dumps(events, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _json_object(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SilentTextLocalizationError("Gemini did not return valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise SilentTextLocalizationError("Gemini response must be a JSON object.")
    return parsed


def _validated_highlight_terms(translation: str, value: object) -> list[str]:
    if not translation or not isinstance(value, list):
        return []
    visible_length = len(re.sub(r"\s+", "", translation))
    budget = max(3, int(math.ceil(visible_length * 0.40)))
    accepted: list[str] = []
    accepted_spans: list[tuple[int, int]] = []
    used = 0
    for raw in value:
        if len(accepted) >= 2 or not isinstance(raw, str):
            break
        term = re.sub(r"\s+", " ", raw.strip())
        if len(term) < 2 or term.casefold() == translation.casefold():
            continue
        match = re.search(re.escape(term), translation, flags=re.IGNORECASE)
        if match is None:
            continue
        span = match.span()
        if any(max(span[0], start) < min(span[1], end) for start, end in accepted_spans):
            continue
        term_size = len(re.sub(r"\s+", "", term))
        if used + term_size > budget:
            continue
        accepted.append(translation[span[0] : span[1]])
        accepted_spans.append(span)
        used += term_size
    return accepted


def validate_gemini_decisions(
    profile: dict[str, Any],
    response: str | dict[str, Any],
    config: object,
) -> dict[str, Any]:
    normalized_config = normalize_localization_config(config)
    payload = _json_object(response)
    items = payload.get("items")
    if not isinstance(items, list):
        raise SilentTextLocalizationError("Gemini response is missing items.")
    event_by_id = {event["track_id"]: event for event in profile["events"]}
    expected = set(event_by_id)
    returned: list[str] = []
    decisions: list[dict[str, Any]] = []
    threshold = normalized_config["confidence_threshold"]
    for raw in items:
        if not isinstance(raw, dict):
            raise SilentTextLocalizationError("A Gemini item is not an object.")
        track_id = raw.get("track_id")
        if not isinstance(track_id, str) or not track_id:
            raise SilentTextLocalizationError("Gemini returned an empty track_id.")
        returned.append(track_id)
        classification = raw.get("classification")
        text_origin = raw.get("text_origin")
        action = raw.get("replacement_action")
        translation = raw.get("vietnamese_translation")
        highlight_terms = raw.get("highlight_terms", [])
        design_role = raw.get("design_role")
        corrected = raw.get("source_text_corrected")
        reason = raw.get("reason_vi")
        confidence = raw.get("confidence")
        if classification not in CLASSIFICATIONS:
            raise SilentTextLocalizationError(
                f"{track_id}: invalid classification {classification!r}."
            )
        if text_origin not in TEXT_ORIGINS:
            raise SilentTextLocalizationError(
                f"{track_id}: invalid text_origin {text_origin!r}."
            )
        if action not in REPLACEMENT_ACTIONS:
            raise SilentTextLocalizationError(
                f"{track_id}: invalid replacement_action {action!r}."
            )
        if design_role is not None and design_role not in DESIGN_ROLES:
            raise SilentTextLocalizationError(
                f"{track_id}: invalid design_role {design_role!r}."
            )
        if not isinstance(translation, str) or not isinstance(corrected, str):
            raise SilentTextLocalizationError(f"{track_id}: invalid text fields.")
        if not isinstance(reason, str) or not reason.strip():
            raise SilentTextLocalizationError(f"{track_id}: missing reason_vi.")
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            raise SilentTextLocalizationError(f"{track_id}: invalid confidence.")
        translation = translation.strip()
        highlight_terms = _validated_highlight_terms(translation, highlight_terms)
        confidence = float(confidence)
        if classification == "spoken_subtitle" and action != "keep_removed":
            raise SilentTextLocalizationError(
                f"{track_id}: spoken subtitle must use keep_removed."
            )
        if classification == "uncertain" and action != "needs_review":
            raise SilentTextLocalizationError(
                f"{track_id}: uncertain text must use needs_review."
            )
        if classification in {"sign_or_ui", "decorative_overlay"} and action != "ignore":
            raise SilentTextLocalizationError(
                f"{track_id}: scene/UI/decorative text must use ignore."
            )
        if (
            classification != "spoken_subtitle"
            and text_origin in {"scene_text", "ui_or_watermark"}
            and action != "ignore"
        ):
            raise SilentTextLocalizationError(
                f"{track_id}: non-overlay text must use ignore."
            )
        if (
            classification != "spoken_subtitle"
            and text_origin == "uncertain"
            and action != "needs_review"
        ):
            raise SilentTextLocalizationError(
                f"{track_id}: uncertain text origin must use needs_review."
            )
        if action == "replace_with_vietnamese":
            if (
                classification not in SILENT_HARD_SUB_CLASSES
                or text_origin != "post_production_overlay"
            ):
                raise SilentTextLocalizationError(
                    f"{track_id}: only silent hard-sub overlays can be replaced."
                )
            if not translation:
                raise SilentTextLocalizationError(
                    f"{track_id}: Vietnamese translation is empty."
                )
        event = event_by_id.get(track_id, {})
        if action == "replace_with_vietnamese":
            if not design_role or design_role == "neutral":
                design_role = infer_design_role(
                    classification=str(classification),
                    translated_text=translation,
                    source_text=corrected,
                    bbox=event.get("bbox"),
                    frame_width=int((profile.get("video") or {}).get("width") or 0),
                    frame_height=int((profile.get("video") or {}).get("height") or 0),
                )
        else:
            design_role = "neutral"
        raw_event_fps = float((profile.get("video") or {}).get("fps") or 0)
        semantic_start_frame = int(event.get("start_frame") or 0)
        semantic_end_frame = int(
            event.get("end_frame") or semantic_start_frame
        )
        if semantic_start_frame > 0 and semantic_end_frame >= semantic_start_frame:
            event_fps = max(raw_event_fps, 1e-6)
            effective_start_frame = int(
                event.get("effective_start_frame") or semantic_start_frame
            )
            effective_end_frame = int(
                event.get("effective_end_frame") or semantic_end_frame
            )
            effective_start_frame = min(
                semantic_start_frame,
                effective_start_frame,
            )
            effective_end_frame = max(semantic_end_frame, effective_end_frame)
            duration_seconds = max(
                0,
                effective_end_frame - effective_start_frame + 1,
            ) / event_fps
            effective_start_ms = int(
                event.get("effective_start_ms")
                if event.get("effective_start_ms") is not None
                else round((effective_start_frame - 1) * 1000 / event_fps)
            )
            effective_end_ms = int(
                event.get("effective_end_ms")
                if event.get("effective_end_ms") is not None
                else round(effective_end_frame * 1000 / event_fps)
            )
        else:
            # Downloaded legacy bundles may only carry millisecond timing.
            # Preserve it verbatim instead of inventing an invalid frame range.
            effective_start_frame = semantic_start_frame
            effective_end_frame = semantic_end_frame
            effective_start_ms = int(event.get("start_ms") or 0)
            effective_end_ms = int(
                event.get("end_ms") or effective_start_ms
            )
            duration_seconds = float(
                event.get("duration_seconds")
                or max(0, effective_end_ms - effective_start_ms) / 1000.0
            )
        # Every approved post-production text event is rendered. The old
        # 0.5-second exception caused valid labels whose OCR track was short to
        # be erased without their Vietnamese replacement.
        eligible_for_auto_translation = True
        auto_approved = bool(
            normalized_config["auto_approve"]
            and action == "replace_with_vietnamese"
            and classification in SILENT_HARD_SUB_CLASSES
            and text_origin == "post_production_overlay"
            and translation
            and confidence >= threshold
        )
        silent_selection_candidate = bool(
            classification in SILENT_HARD_SUB_CLASSES
            and text_origin == "post_production_overlay"
            and confidence >= threshold
        )
        selected_for_inpaint = bool(
            classification == "spoken_subtitle"
            or (
                silent_selection_candidate
                and action == "replace_with_vietnamese"
                and (
                    not normalized_config["auto_render"]
                    or auto_approved
                )
            )
        )
        if classification == "spoken_subtitle":
            render_action = "remove_original"
        elif auto_approved:
            render_action = "replace_with_vietnamese"
        else:
            # VSR already generated a clean video. The postprocessor restores
            # this event from the source video so low-confidence text is not lost.
            render_action = "restore_original"
        decisions.append(
            {
                "track_id": track_id,
                "classification": classification,
                "text_origin": text_origin,
                "source_text_corrected": corrected.strip(),
                "vietnamese_translation": translation,
                "highlight_terms": (
                    highlight_terms if action == "replace_with_vietnamese" else []
                ),
                "design_role": design_role,
                "replacement_action": action,
                "confidence": confidence,
                "reason_vi": reason.strip(),
                "auto_approved": auto_approved,
                "selected_for_inpaint": selected_for_inpaint,
                "render_action": render_action,
                "start_ms": effective_start_ms,
                "end_ms": effective_end_ms,
                "start_frame": effective_start_frame,
                "end_frame": effective_end_frame,
                "semantic_start_frame": semantic_start_frame,
                "semantic_end_frame": semantic_end_frame,
                "duration_seconds": round(duration_seconds, 6),
                "eligible_for_auto_translation": eligible_for_auto_translation,
                "bbox": event.get("bbox") or [],
                "bbox_normalized": event.get("bbox_normalized") or [],
                "visual_style": event.get("visual_style") or {},
                "persistent_fixed_overlay": bool(event.get("persistent_fixed_overlay")),
                "persistent_bbox": list(event.get("persistent_bbox") or []),
                "persistent_visual_supertrack": bool(
                    event.get("persistent_visual_supertrack")
                ),
                "persistent_visual_child_count": int(
                    event.get("persistent_visual_child_count") or 0
                ),
            }
        )
    if len(returned) != len(set(returned)):
        raise SilentTextLocalizationError("Gemini returned duplicate track_id values.")
    actual = set(returned)
    if actual != expected:
        raise SilentTextLocalizationError(
            "Gemini track_id set mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    decisions.sort(
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
            item["track_id"],
        )
    )
    restore_count = sum(
        item["render_action"] == "restore_original" for item in decisions
    )
    summary = {
        "event_count": len(decisions),
        "selected_for_inpaint_count": sum(
            item["selected_for_inpaint"] for item in decisions
        ),
        "spoken_remove_count": sum(
            item["classification"] == "spoken_subtitle"
            and item["render_action"] == "remove_original"
            for item in decisions
        ),
        "auto_approved_count": sum(item["auto_approved"] for item in decisions),
        "replace_count": sum(
            item["render_action"] == "replace_with_vietnamese"
            for item in decisions
        ),
        "remove_count": sum(
            item["render_action"] == "remove_original" for item in decisions
        ),
        "restore_count": restore_count,
        "semantic_keep_decision_count": restore_count,
    }
    return {
        "schema": REVIEW_SCHEMA,
        "version": REVIEW_VERSION,
        "source_video_name": profile.get("source_video_name") or "",
        "config": normalized_config,
        "summary": summary,
        "decisions": decisions,
    }


def _box_overlap_for_selection(first: list[int], second: list[int]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    left = max(int(first[0]), int(second[0]))
    right = min(int(first[1]), int(second[1]))
    top = max(int(first[2]), int(second[2]))
    bottom = min(int(first[3]), int(second[3]))
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(
        1,
        (int(first[1]) - int(first[0])) * (int(first[3]) - int(first[2])),
    )
    second_area = max(
        1,
        (int(second[1]) - int(second[0])) * (int(second[3]) - int(second[2])),
    )
    return intersection / max(1, min(first_area, second_area))


def _box_covers_target(first: list[int], target: list[int]) -> float:
    """Return how much of ``target`` is covered by ``first``.

    Persistent-overlay backfill needs directional coverage.  The symmetric
    overlap helper above treats a small nested line as a perfect match for a
    larger multi-line envelope, which can incorrectly suppress the envelope.
    """
    if len(first) != 4 or len(target) != 4:
        return 0.0
    left = max(int(first[0]), int(target[0]))
    right = min(int(first[1]), int(target[1]))
    top = max(int(first[2]), int(target[2]))
    bottom = min(int(first[3]), int(target[3]))
    intersection = max(0, right - left) * max(0, bottom - top)
    target_area = max(
        1,
        (int(target[1]) - int(target[0]))
        * (int(target[3]) - int(target[2])),
    )
    return intersection / target_area


def _enforce_spoken_subtitle_erase_invariant(review: dict[str, Any]) -> set[str]:
    """Keep spoken subtitles on the exact-mask-only erase path.

    Spoken subtitles are always removal decisions, but semantic classification
    does not grant permission to invent pixels outside OCR evidence. Exact mask
    ownership is checked by ``filter_subtitle_frames_for_review`` before STTN.
    """
    spoken_ids: set[str] = set()
    for decision in review.get("decisions") or []:
        if str(decision.get("classification") or "") != "spoken_subtitle":
            continue
        track_id = str(decision.get("track_id") or "")
        if not track_id:
            continue
        spoken_ids.add(track_id)
        decision["selected_for_inpaint"] = True
        decision["render_action"] = "remove_original"
        decision["replacement_action"] = "keep_removed"
        decision["vietnamese_translation"] = ""
        decision["auto_approved"] = False
        if decision.get("mapping_guard") == "preserve_original_no_exact_mask":
            decision.pop("mapping_guard", None)
        decision["erase_policy"] = "exact_frame_masks_only"
    return spoken_ids


def build_persistent_editor_mask_clip_specs(
    review: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """Return narrowly-scoped ROI exceptions for confirmed persistent editor masks.

    Normal OCR/STTN masks remain clipped to the user/reviewed ROI.  A persistent
    semantic super-track may cross that boundary only when its render anchor was
    strongly classified as a post-production overlay.  Even then, the allowed
    exception is capped by the same edge-microtext padding policy that created
    ``persistent_mask_bbox``.

    Returned ROIs use ``(ymin, ymax, xmin, xmax)`` to match ``sub_areas``.
    """
    width = max(0, int(frame_width))
    height = max(0, int(frame_height))
    if width <= 0 or height <= 0:
        return []
    video = {"width": width, "height": height}
    specs: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for decision in review.get("decisions") or []:
        if not (
            bool(decision.get("selected_for_inpaint"))
            and bool(decision.get("persistent_fixed_overlay"))
            and bool(decision.get("render_anchor_for_semantic"))
            and str(decision.get("semantic_review_track_id") or "")
            and decision.get("text_origin") == "post_production_overlay"
            and decision.get("render_action") == "replace_with_vietnamese"
            and float(decision.get("confidence") or 0.0) >= 0.90
        ):
            continue
        envelope = [int(value) for value in decision.get("persistent_bbox") or []]
        requested = [
            int(value) for value in decision.get("persistent_mask_bbox") or []
        ]
        if len(envelope) != 4 or len(requested) != 4:
            continue
        if not _edge_microtext_box(envelope, video):
            continue
        safe_limit = _persistent_edge_mask_bbox(envelope, video)
        if len(safe_limit) != 4:
            continue
        # Never trust a stale/oversized review bundle to widen the mask farther
        # than the current microtext policy permits.
        clipped = [
            max(safe_limit[0], requested[0]),
            min(safe_limit[1], requested[1]),
            max(safe_limit[2], requested[2]),
            min(safe_limit[3], requested[3]),
        ]
        if (
            clipped[1] <= clipped[0]
            or clipped[3] <= clipped[2]
            or _box_covers_target(clipped, envelope) < 0.99
        ):
            continue
        start_frame = max(1, int(decision.get("start_frame") or 1))
        end_frame = max(
            start_frame,
            int(decision.get("end_frame") or start_frame),
        )
        specs.append(
            (
                start_frame,
                end_frame,
                (clipped[2], clipped[3], clipped[0], clipped[1]),
            )
        )
    return specs


def filter_subtitle_frames_for_review(
    subtitle_frames: dict[int, list[list[int] | tuple[int, int, int, int]]],
    review: dict[str, Any],
    *,
    fps: float,
    profile: dict[str, Any] | None = None,
) -> dict[int, list[list[int] | tuple[int, int, int, int]]]:
    """Apply decisions through exact ``track_id`` frame/box observations.

    New profiles must cover every mask that can reach STTN.  A missing or extra
    observation is a contract error, not permission to erase an unknown region.
    Legacy profiles without ``frame_boxes`` retain the old overlap matcher so
    existing downloaded review bundles remain usable.
    """
    spoken_subtitle_ids = _enforce_spoken_subtitle_erase_invariant(review)
    mask_mapping = (profile or {}).get("mask_mapping") or {}
    exact_tracks = mask_mapping.get("exact_tracks")
    if profile is not None and isinstance(exact_tracks, list):
        decision_by_id = {
            str(item.get("track_id") or ""): item
            for item in review.get("decisions") or []
        }
        expected_ids = {
            str(event.get("track_id") or "")
            for event in profile.get("events") or []
        }
        if set(decision_by_id) != expected_ids:
            raise SilentTextLocalizationError(
                "Review/profile semantic track_id mismatch while selecting STTN masks."
            )
        original = {
            (int(frame_no), tuple(int(value) for value in box))
            for frame_no, boxes in subtitle_frames.items()
            for box in boxes
        }
        observed: set[tuple[int, tuple[int, int, int, int]]] = set()
        selected: set[tuple[int, tuple[int, int, int, int]]] = set()
        selected_semantic_ids = {
            track_id
            for track_id, decision in decision_by_id.items()
            if bool(decision.get("selected_for_inpaint"))
        }
        mapped_selected_semantic_ids: set[str] = set()
        selected_exact_mask_track_count = 0
        valid_statuses = {"assigned", "ambiguous", "unowned"}
        seen_mask_track_ids: set[str] = set()
        for exact_track in exact_tracks:
            if not isinstance(exact_track, dict):
                raise SilentTextLocalizationError("Invalid exact mask track mapping.")
            mask_track_id = str(exact_track.get("mask_track_id") or "")
            status = str(exact_track.get("status") or "")
            owner_track_id = exact_track.get("owner_track_id")
            if (
                not mask_track_id
                or mask_track_id in seen_mask_track_ids
                or status not in valid_statuses
            ):
                raise SilentTextLocalizationError(
                    "Invalid or duplicate exact mask track identity."
                )
            seen_mask_track_ids.add(mask_track_id)
            if status == "assigned":
                if owner_track_id not in expected_ids:
                    raise SilentTextLocalizationError(
                        f"{mask_track_id}: assigned mask has no semantic owner."
                    )
                choose = bool(
                    decision_by_id[str(owner_track_id)].get(
                        "selected_for_inpaint"
                    )
                )
                if choose:
                    mapped_selected_semantic_ids.add(str(owner_track_id))
            elif status == "ambiguous":
                if owner_track_id not in {None, ""}:
                    raise SilentTextLocalizationError(
                        f"{mask_track_id}: unsafe owner on {status} mask."
                    )
                candidate_ids = list(
                    dict.fromkeys(
                        str(track_id)
                        for track_id in exact_track.get("candidate_track_ids") or []
                        if str(track_id)
                    )
                )
                try:
                    candidate_count = int(exact_track.get("candidate_count"))
                except (TypeError, ValueError):
                    candidate_count = 0
                candidates_are_complete = (
                    len(candidate_ids) >= 2
                    and candidate_count == len(candidate_ids)
                    and set(candidate_ids).issubset(expected_ids)
                )
                ambiguous_spoken_ids = set(candidate_ids) & spoken_subtitle_ids
                all_candidates_are_spoken = bool(candidate_ids) and set(
                    candidate_ids
                ).issubset(spoken_subtitle_ids)
                all_candidates_selected = candidates_are_complete and all(
                    bool(decision_by_id[track_id].get("selected_for_inpaint"))
                    for track_id in candidate_ids
                )
                # One exact OCR mask can legitimately overlap several semantic
                # fragments of the same spoken subtitle. When every complete
                # candidate independently agrees on removal, selecting that
                # observed mask does not invent any pixel geometry. Mixed
                # spoken/editor ambiguity remains unsafe and fails below.
                choose = all_candidates_selected and (
                    not ambiguous_spoken_ids or all_candidates_are_spoken
                )
                contained_selected_ids: set[str] = set()
                if candidates_are_complete and not choose:
                    frame_boxes = [
                        item
                        for item in exact_track.get("frame_boxes") or []
                        if isinstance(item, dict)
                        and len(item.get("bbox") or []) == 4
                    ]
                    for candidate_id in candidate_ids:
                        if candidate_id in spoken_subtitle_ids:
                            continue
                        candidate_decision = decision_by_id[candidate_id]
                        if not (
                            bool(candidate_decision.get("selected_for_inpaint"))
                            and candidate_decision.get("text_origin")
                            == "post_production_overlay"
                            and candidate_decision.get("render_action")
                            in {"remove_original", "replace_with_vietnamese"}
                            and float(candidate_decision.get("confidence") or 0.0) >= 0.90
                        ):
                            continue
                        candidate_box = [
                            int(value) for value in candidate_decision.get("bbox") or []
                        ]
                        if len(candidate_box) != 4 or not frame_boxes:
                            continue
                        candidate_area = max(
                            1,
                            (candidate_box[1] - candidate_box[0])
                            * (candidate_box[3] - candidate_box[2]),
                        )
                        safely_related = True
                        for item in frame_boxes:
                            exact_box = [
                                int(value) for value in item.get("bbox") or []
                            ]
                            exact_area = max(
                                1,
                                (exact_box[1] - exact_box[0])
                                * (exact_box[3] - exact_box[2]),
                            )
                            candidate_contains_exact = (
                                _box_covers_target(candidate_box, exact_box) >= 0.90
                            )
                            exact_contains_candidate = bool(
                                exact_area <= candidate_area * 2.5
                                and _box_covers_target(exact_box, candidate_box) >= 0.90
                                and _review_box_affinity(exact_box, candidate_box) >= 0.90
                            )
                            if not (candidate_contains_exact or exact_contains_candidate):
                                safely_related = False
                                break
                        if safely_related:
                            contained_selected_ids.add(candidate_id)
                    choose = bool(contained_selected_ids)
                if choose:
                    if contained_selected_ids:
                        mapped_selected_semantic_ids.update(contained_selected_ids)
                    else:
                        mapped_selected_semantic_ids.update(candidate_ids)
            else:
                if owner_track_id not in {None, ""}:
                    raise SilentTextLocalizationError(
                        f"{mask_track_id}: unsafe owner on {status} mask."
                    )
                # Fail closed: an unowned final mask remains in the original
                # video and never reaches STTN.
                choose = False
            if choose:
                selected_exact_mask_track_count += 1
            for item in exact_track.get("frame_boxes") or []:
                box = tuple(int(value) for value in item.get("bbox") or [])
                if len(box) != 4:
                    raise SilentTextLocalizationError(
                        f"{mask_track_id}: invalid exact frame box."
                    )
                key = (int(item.get("frame_no") or 0), box)
                if key in observed:
                    raise SilentTextLocalizationError(
                        f"{mask_track_id}: duplicate exact frame box ownership."
                    )
                observed.add(key)
                if choose:
                    selected.add(key)
        if observed != original:
            missing = len(original - observed)
            extra = len(observed - original)
            raise SilentTextLocalizationError(
                "Exact mask mapping does not cover the STTN mask timeline "
                f"(missing={missing}, extra={extra})."
            )
        multiline_shared_mask_covered_ids: set[str] = set()
        for track_id in selected_semantic_ids - mapped_selected_semantic_ids:
            anchor = decision_by_id[track_id]
            if not (
                anchor.get("render_action") == "replace_with_vietnamese"
                and anchor.get("text_origin") == "post_production_overlay"
                and anchor.get("multiline_render_group") == track_id
                and anchor.get("multiline_render_row") == 0
            ):
                continue
            bridge = next((
                item for item in decision_by_id.values()
                if item.get("multiline_render_group") == track_id
                and item.get("multiline_render_row") == 1
                and bool(item.get("selected_for_inpaint"))
                and item.get("render_action") == "replace_with_vietnamese"
            ), None)
            tail = next((
                item for item in decision_by_id.values()
                if item.get("render_anchor_track_id") == track_id
                and item.get("routing_override")
                == "persistent_multiline_duplicate_fragment"
                and bool(item.get("selected_for_inpaint"))
                and item.get("render_action") == "remove_original"
            ), None)
            if bridge is None or tail is None:
                continue
            row_box = [int(value) for value in anchor.get("render_target_bbox") or []]
            bridge_box = [int(value) for value in bridge.get("render_target_bbox") or []]
            if (
                len(row_box) != 4 or len(bridge_box) != 4
                or row_box[3] > bridge_box[2]
                or row_box[0] >= row_box[1]
                or row_box[2] >= row_box[3]
            ):
                continue
            start_frame = max(1, int(anchor.get("start_frame") or 0))
            end_frame = int(anchor.get("end_frame") or 0)
            if end_frame < start_frame:
                continue
            covered_frames: set[int] = set()
            covered_owners: set[str] = set()
            companion_ids = {str(bridge["track_id"]), str(tail["track_id"])}
            for exact_track in exact_tracks:
                owner = str(exact_track.get("owner_track_id") or "")
                if exact_track.get("status") != "assigned" or owner not in companion_ids:
                    continue
                for item in exact_track.get("frame_boxes") or []:
                    frame_no = int(item.get("frame_no") or 0)
                    box = [int(value) for value in item.get("bbox") or []]
                    if (
                        len(box) == 4
                        and (frame_no, tuple(box)) in selected
                        and _box_covers_target(box, row_box) >= 0.90
                    ):
                        covered_frames.add(frame_no)
                        covered_owners.add(owner)
            if (
                covered_owners == companion_ids
                and all(frame in covered_frames
                        for frame in range(start_frame, end_frame + 1))
            ):
                mapped_selected_semantic_ids.add(track_id)
                multiline_shared_mask_covered_ids.add(track_id)
                anchor["mapping_guard"] = "multiline_shared_exact_mask_coverage"
        persistent_anchor_covered_ids: set[str] = set()
        for track_id in selected_semantic_ids - mapped_selected_semantic_ids:
            decision = decision_by_id[track_id]
            route = decision.get("routing_override")
            if route in {
                "persistent_edge_fragment_inherit",
                "persistent_edge_nested_render_suppressed",
            }:
                anchor_id = str(decision.get("linked_text_track_id") or "")
            elif route == "semantic_child_fragment" and bool(
                decision.get("persistent_fixed_overlay")
            ):
                anchor_id = str(decision.get("render_anchor_track_id") or "")
            else:
                continue
            anchor = decision_by_id.get(anchor_id)
            if not anchor:
                continue
            if not (
                bool(anchor.get("selected_for_inpaint"))
                and bool(anchor.get("persistent_fixed_overlay"))
                and bool(anchor.get("render_anchor_for_semantic"))
                and anchor.get("render_action") == "replace_with_vietnamese"
            ):
                continue
            if route == "semantic_child_fragment":
                semantic_id = str(decision.get("semantic_review_track_id") or "")
                if not semantic_id or semantic_id != str(
                    anchor.get("semantic_review_track_id") or ""
                ):
                    continue
            anchor_box = [int(value) for value in anchor.get("persistent_bbox") or []]
            child_box = [int(value) for value in decision.get("bbox") or []]
            if len(anchor_box) != 4 or len(child_box) != 4:
                continue
            if _box_covers_target(anchor_box, child_box) < 0.80:
                continue
            child_start = max(1, int(decision.get("start_frame") or 1))
            child_end = max(child_start, int(decision.get("end_frame") or child_start))
            anchor_start = max(1, int(anchor.get("start_frame") or 1))
            anchor_end = max(anchor_start, int(anchor.get("end_frame") or anchor_start))
            if child_end < anchor_start or child_start > anchor_end:
                continue
            persistent_anchor_covered_ids.add(track_id)
            decision["mapping_guard"] = "persistent_anchor_envelope_backfill"
        spoken_unmapped_ids = sorted(
            selected_semantic_ids
            - mapped_selected_semantic_ids
            - persistent_anchor_covered_ids
            & spoken_subtitle_ids
        )
        if spoken_unmapped_ids:
            raise SilentTextLocalizationError(
                "Spoken subtitles require assigned exact masks before STTN: "
                + ", ".join(spoken_unmapped_ids)
            )
        unmapped_selected_ids = sorted(
            selected_semantic_ids
            - mapped_selected_semantic_ids
            - persistent_anchor_covered_ids
            - spoken_subtitle_ids
        )
        # An ambiguous/unmapped semantic event cannot safely drive STTN and
        # cannot be translated on top of its still-visible original text.
        # Preserve it explicitly before the expensive inpaint pass instead of
        # failing after the final render or creating duplicate bilingual text.
        for track_id in unmapped_selected_ids:
            decision = decision_by_id[track_id]
            decision["selected_for_inpaint"] = False
            decision["render_action"] = "restore_original"
            decision["mapping_guard"] = "preserve_original_no_exact_mask"
        if unmapped_selected_ids:
            review["summary"] = _review_summary(list(decision_by_id.values()))
        review["mask_mapping_coverage"] = {
            "mode": "semantic_event_to_exact_mask_graph",
            "requested_selected_semantic_event_count": len(selected_semantic_ids),
            "selected_semantic_event_count": len(
                selected_semantic_ids - set(unmapped_selected_ids)
            ),
            "mapped_selected_semantic_event_count": len(
                mapped_selected_semantic_ids
            ),
            "preserved_unmapped_semantic_event_count": len(
                unmapped_selected_ids
            ),
            "preserved_unmapped_semantic_event_ids": unmapped_selected_ids,
            "unmapped_selected_semantic_event_ids": [],
            "selected_exact_mask_track_count": selected_exact_mask_track_count,
            "selected_exact_mask_box_count": len(selected),
            "complete": True,
            "requested_mapping_complete": not unmapped_selected_ids,
            "guard_policy": "preserve_original_before_sttn_when_unmapped",
        }
        if multiline_shared_mask_covered_ids:
            review["mask_mapping_coverage"]["multiline_shared_mask_covered_ids"] = sorted(
                multiline_shared_mask_covered_ids
            )
        if persistent_anchor_covered_ids:
            review["mask_mapping_coverage"].update(
                {
                    "persistent_anchor_covered_semantic_event_count": len(
                        persistent_anchor_covered_ids
                    ),
                    "persistent_anchor_covered_semantic_event_ids": sorted(
                        persistent_anchor_covered_ids
                    ),
                    "guard_policy": (
                        "preserve_original_before_sttn_when_unmapped_except_confirmed_"
                        "persistent_anchor_envelope"
                    ),
                }
            )
        filtered: dict[int, list[list[int] | tuple[int, int, int, int]]] = {}
        for frame_no, boxes in subtitle_frames.items():
            kept = [
                box
                for box in boxes
                if (int(frame_no), tuple(int(value) for value in box)) in selected
            ]
            if kept:
                filtered[int(frame_no)] = kept
        backfilled_box_count = 0
        for track_id, decision in decision_by_id.items():
            if track_id in spoken_subtitle_ids:
                continue
            if not (
                decision.get("persistent_fixed_overlay")
                and decision.get("render_anchor_for_semantic")
                and decision.get("selected_for_inpaint")
            ):
                continue
            bbox = [
                int(value)
                for value in (
                    decision.get("persistent_mask_bbox")
                    or decision.get("persistent_bbox")
                    or []
                )
            ]
            if len(bbox) != 4:
                continue
            start_frame = max(1, int(decision.get("start_frame") or 1))
            end_frame = max(start_frame, int(decision.get("end_frame") or start_frame))
            for frame_no in range(start_frame, end_frame + 1):
                boxes = filtered.setdefault(frame_no, [])
                if any(_box_covers_target(list(box), bbox) >= 0.90 for box in boxes):
                    continue
                boxes.append(list(bbox))
                backfilled_box_count += 1
        if backfilled_box_count:
            review["mask_mapping_coverage"]["persistent_backfilled_box_count"] = backfilled_box_count
        return filtered

    if profile is not None and all(
        isinstance(event.get("frame_boxes"), list)
        for event in profile.get("events") or []
    ):
        decision_by_id = {
            str(item.get("track_id") or ""): item
            for item in review.get("decisions") or []
        }
        expected_ids = {
            str(event.get("track_id") or "")
            for event in profile.get("events") or []
        }
        if set(decision_by_id) != expected_ids:
            raise SilentTextLocalizationError(
                "Review/profile track_id mismatch while selecting STTN masks."
            )

        original = {
            (int(frame_no), tuple(int(value) for value in box))
            for frame_no, boxes in subtitle_frames.items()
            for box in boxes
        }
        observed: set[tuple[int, tuple[int, int, int, int]]] = set()
        selected: set[tuple[int, tuple[int, int, int, int]]] = set()
        for event in profile.get("events") or []:
            track_id = str(event.get("track_id") or "")
            choose = bool(decision_by_id[track_id].get("selected_for_inpaint"))
            for item in event.get("frame_boxes") or []:
                box = tuple(int(value) for value in item.get("bbox") or [])
                if len(box) != 4:
                    raise SilentTextLocalizationError(
                        f"{track_id}: invalid exact frame box."
                    )
                key = (int(item.get("frame_no") or 0), box)
                if key in observed:
                    raise SilentTextLocalizationError(
                        f"{track_id}: duplicate exact frame box ownership."
                    )
                observed.add(key)
                if choose:
                    selected.add(key)
        if observed != original:
            missing = len(original - observed)
            extra = len(observed - original)
            raise SilentTextLocalizationError(
                "Subtitle event graph does not exactly cover the STTN mask "
                f"timeline (missing={missing}, extra={extra})."
            )
        filtered: dict[int, list[list[int] | tuple[int, int, int, int]]] = {}
        for frame_no, boxes in subtitle_frames.items():
            kept = [
                box
                for box in boxes
                if (int(frame_no), tuple(int(value) for value in box)) in selected
            ]
            if kept:
                filtered[int(frame_no)] = kept
        return filtered

    # Legacy downloaded bundle compatibility.
    effective_fps = max(float(fps), 1e-6)
    selected = []
    reviewed = []
    for item in review.get("decisions") or []:
        box = item.get("bbox") or []
        if not isinstance(box, list) or len(box) != 4:
            continue
        start_frame = max(
            1,
            int(
                math.floor(
                    float(item.get("start_ms") or 0) * effective_fps / 1000
                )
            )
            + 1,
        )
        end_frame = max(
            start_frame,
            int(
                math.ceil(
                    float(item.get("end_ms") or 0) * effective_fps / 1000
                )
            ),
        )
        target = (start_frame, end_frame, [int(value) for value in box])
        reviewed.append(target)
        if item.get("selected_for_inpaint"):
            selected.append(target)

    filtered: dict[int, list[list[int] | tuple[int, int, int, int]]] = {}
    for raw_frame_no, boxes in subtitle_frames.items():
        frame_no = int(raw_frame_no)
        kept = []
        for box in boxes:
            candidate = [int(value) for value in box]
            selected_match = any(
                start <= frame_no <= end
                and _box_overlap_for_selection(candidate, target) >= 0.25
                for start, end, target in selected
            )
            reviewed_match = any(
                start <= frame_no <= end
                and _box_overlap_for_selection(candidate, target) >= 0.25
                for start, end, target in reviewed
            )
            if selected_match or not reviewed_match:
                kept.append(box)
        if kept:
            filtered[frame_no] = kept
    return filtered


def write_review(review: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _safe_asset(profile_path: Path, reference: object) -> Path:
    root = profile_path.parent.resolve()
    reference_path = Path(str(reference or ""))
    if reference_path.is_absolute() or ".." in reference_path.parts:
        raise SilentTextLocalizationError(
            f"Unsafe subtitle event asset reference: {reference!r}"
        )
    candidate = (root / reference_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SilentTextLocalizationError(
            f"Subtitle event asset escapes bundle: {reference!r}"
        ) from exc
    if not candidate.is_file():
        raise SilentTextLocalizationError(
            f"Subtitle event asset is missing: {reference!r}"
        )
    return candidate


def _fit_image(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(
        image,
        ((width - image.width) // 2, (height - image.height) // 2),
    )
    return canvas


def build_contact_sheets(
    profile: dict[str, Any],
    profile_path: str | Path,
    output_folder: str | Path,
) -> list[Path]:
    source = Path(profile_path)
    output = Path(output_folder)
    output.mkdir(parents=True, exist_ok=True)
    title_font = _font(23)
    small_font = _font(17)
    sheets: list[Path] = []
    for page_index in range(math.ceil(len(profile["events"]) / CONTACT_SHEET_EVENTS)):
        events = profile["events"][
            page_index * CONTACT_SHEET_EVENTS : (page_index + 1) * CONTACT_SHEET_EVENTS
        ]
        sheet = Image.new("RGB", CONTACT_SHEET_SIZE, "#181818")
        draw = ImageDraw.Draw(sheet)
        for cell_index, event in enumerate(events):
            column = cell_index % 2
            row = cell_index // 2
            x, y = column * CONTACT_CELL_SIZE[0], row * CONTACT_CELL_SIZE[1]
            frame_path = _safe_asset(source, event.get("representative_frame"))
            frame = _fit_image(frame_path, 732, 292)
            sheet.paste(frame, (x + 12, y + 72))
            draw.text(
                (x + 12, y + 10),
                (
                    f"{event['track_id']}  "
                    f"{int(event.get('start_ms') or 0) / 1000:.2f}s–"
                    f"{int(event.get('end_ms') or 0) / 1000:.2f}s"
                ),
                fill="white",
                font=title_font,
            )
            draw.text(
                (x + 12, y + 370),
                f"OCR: {str(event.get('ocr_best_text') or '')[:54]}",
                fill="#ffd966",
                font=small_font,
            )
            draw.text(
                (x + 360, y + 370),
                (
                    f"conf={float(event.get('ocr_best_confidence') or 0):.3f} "
                    f"bbox={event.get('bbox') or []}"
                ),
                fill="#c8c8c8",
                font=small_font,
            )
            draw.rectangle(
                (
                    x,
                    y,
                    x + CONTACT_CELL_SIZE[0] - 1,
                    y + CONTACT_CELL_SIZE[1] - 1,
                ),
                outline="#555555",
                width=2,
            )
        path = output / f"silent-text-contact-{page_index + 1:03d}.jpg"
        sheet.save(path, "JPEG", quality=90, optimize=True)
        sheets.append(path)
    return sheets


def discover_srt(video_path: str | Path) -> Path | None:
    video = Path(video_path)
    for suffix in (".srt", ".SRT"):
        candidate = video.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def load_uploaded_api_keys(
    path: str | Path,
    *,
    expected_sha256: str = "",
) -> list[tuple[str, str]]:
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise SilentTextLocalizationError(
            "Uploaded Gemini credential bundle is missing or unreadable."
        ) from exc
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise SilentTextLocalizationError(
            "Uploaded Gemini credential bundle fingerprint does not match."
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SilentTextLocalizationError(
            "Uploaded Gemini credential bundle is invalid."
        ) from exc
    keys = payload.get("keys") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != API_KEY_BUNDLE_SCHEMA
        or payload.get("version") != API_KEY_BUNDLE_VERSION
        or not isinstance(keys, list)
    ):
        raise SilentTextLocalizationError(
            "Uploaded Gemini credential bundle has an unsupported schema."
        )
    available: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in keys:
        key = str(value or "").strip()
        if not key or key in seen or any(character.isspace() for character in key):
            raise SilentTextLocalizationError(
                "Uploaded Gemini credential bundle contains an invalid or duplicate key."
            )
        seen.add(key)
        available.append((f"uploaded_key_{len(available) + 1}", key))
    if not 1 <= len(available) <= 100:
        raise SilentTextLocalizationError(
            "Uploaded Gemini credential bundle must contain 1-100 keys."
        )
    return available


def kaggle_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        value = str(UserSecretsClient().get_secret(name) or "").strip()
    except Exception as exc:
        raise SilentTextLocalizationError(
            f"Kaggle Secret {name} is missing or unavailable."
        ) from exc
    if not value:
        raise SilentTextLocalizationError(f"Kaggle Secret {name} is empty.")
    return value


def kaggle_secret_pool(names: list[str] | tuple[str, ...]) -> list[tuple[str, str]]:
    available: list[tuple[str, str]] = []
    for name in names:
        try:
            value = kaggle_secret(str(name))
        except SilentTextLocalizationError:
            continue
        available.append((str(name), value))
    if not available:
        labels = ", ".join(str(name) for name in names)
        raise SilentTextLocalizationError(
            f"No configured Kaggle Gemini Secret is available ({labels})."
        )
    return available


def gemini_api_key_pool(config: dict[str, Any]) -> list[tuple[str, str]]:
    key_file_path = str(config.get("api_key_file_path") or "").strip()
    if key_file_path:
        return load_uploaded_api_keys(
            key_file_path,
            expected_sha256=str(config.get("api_key_file_sha256") or ""),
        )
    return kaggle_secret_pool(config["api_key_secrets"])


def _quota_dimension(message: str) -> str:
    text = str(message or "").lower()
    if any(marker in text for marker in ("requests per day", "per_day", "perday", "rpd", "daily")):
        return "RPD"
    if any(marker in text for marker in ("tokens per minute", "tpm")):
        return "TPM"
    if any(marker in text for marker in ("requests per minute", "rpm", "per_minute", "perminute")):
        return "RPM"
    if "spend" in text:
        return "SPEND"
    return "UNKNOWN"


def _retry_after_seconds(message: str) -> float | None:
    for pattern in (
        r"retryDelay['\"\s:]+(\d+(?:\.\d+)?)s",
        r"retry\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*s",
    ):
        match = re.search(pattern, str(message or ""), flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _response_error_detail(response: object) -> str:
    values: list[str] = []
    try:
        payload = response.json()
    except Exception:
        payload = None
    if payload is not None:
        try:
            values.append(json.dumps(payload, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            pass
    text_value = str(getattr(response, "text", "") or "").strip()
    if text_value and not values:
        values.append(text_value[:4000])
    return "\n".join(values)


class GeminiRestClient:
    """Small Files API + generateContent client with streaming media upload."""

    def __init__(
        self,
        api_key: str,
        *,
        session: object | None = None,
        timeout_seconds: int = 180,
        log: Callable[[str], None] | None = None,
    ) -> None:
        import requests

        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.log = log or (lambda _message: None)

    def _request(self, method: str, url: str, **kwargs):
        import requests

        headers = dict(kwargs.pop("headers", {}))
        headers["x-goog-api-key"] = self.api_key
        stream = kwargs.get("data")
        stream_position = None
        if hasattr(stream, "tell") and hasattr(stream, "seek"):
            try:
                stream_position = stream.tell()
            except (OSError, ValueError):
                stream_position = None
        request_timeout_seconds = int(
            kwargs.pop("request_timeout_seconds", self.timeout_seconds)
        )
        request_attempts = max(1, int(kwargs.pop("request_attempts", 3)))
        last_status: int | None = None
        for attempt in range(request_attempts):
            try:
                if attempt and stream_position is not None:
                    stream.seek(stream_position)
                response = self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=request_timeout_seconds,
                    **kwargs,
                )
                last_status = int(response.status_code)
                detail = _response_error_detail(response)
                if response.status_code in {401, 403}:
                    raise GeminiRequestError(
                        f"Gemini key/project returned HTTP {response.status_code}.",
                        status_code=response.status_code,
                        rotate_key=True,
                    )
                if response.status_code == 404:
                    raise GeminiRequestError(
                        "Gemini resource/model returned HTTP 404.",
                        status_code=404,
                        rotate_key=False,
                    )
                if response.status_code == 429:
                    retry_after = None
                    try:
                        retry_after_header = response.headers.get("Retry-After")
                    except Exception:
                        retry_after_header = None
                    if retry_after_header:
                        try:
                            retry_after = max(0.0, float(retry_after_header))
                        except (TypeError, ValueError):
                            retry_after = None
                    if retry_after is None:
                        retry_after = _retry_after_seconds(detail)
                    raise GeminiRequestError(
                        "Gemini project returned HTTP 429.",
                        status_code=429,
                        rotate_key=True,
                        quota_dimension=_quota_dimension(detail),
                        retry_after_seconds=retry_after,
                    )
                if 400 <= response.status_code < 500:
                    raise GeminiRequestError(
                        f"Gemini request was rejected with HTTP {response.status_code}.",
                        status_code=response.status_code,
                        rotate_key=False,
                    )
                if response.status_code >= 500:
                    if attempt == request_attempts - 1:
                        raise GeminiRequestError(
                            f"Gemini temporary HTTP {response.status_code} after retries.",
                            status_code=response.status_code,
                            rotate_key=response.status_code != 503,
                        )
                    delay = min(30, 2 ** attempt)
                    self.log(f"Gemini temporary HTTP {response.status_code}; retrying in {delay}s.")
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                return response
            except GeminiRequestError:
                raise
            except requests.RequestException:
                if attempt == request_attempts - 1:
                    break
                delay = min(30, 2 ** attempt)
                self.log(f"Gemini temporary error; retrying in {delay}s.")
                time.sleep(delay)
        raise GeminiRequestError(
            "Gemini transport failed after retries.",
            status_code=last_status,
            rotate_key=True,
        )

    def upload_file(self, path: str | Path) -> dict[str, Any]:
        source = Path(path)
        mime_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        size = source.stat().st_size
        start = self._request(
            "POST",
            f"{GEMINI_API_ROOT}/upload/v1beta/files",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(size),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": source.name}},
        )
        upload_url = start.headers.get("X-Goog-Upload-URL")
        if not upload_url:
            raise SilentTextLocalizationError(
                "Gemini Files API did not return a resumable upload URL."
            )
        with source.open("rb") as stream:
            uploaded = self._request(
                "POST",
                upload_url,
                headers={
                    "Content-Length": str(size),
                    "X-Goog-Upload-Offset": "0",
                    "X-Goog-Upload-Command": "upload, finalize",
                },
                data=stream,
            )
        payload = uploaded.json()
        file_info = payload.get("file") if isinstance(payload, dict) else None
        if not isinstance(file_info, dict):
            raise SilentTextLocalizationError(
                "Gemini Files API returned an invalid file object."
            )
        return self.wait_until_active(file_info)

    def wait_until_active(self, file_info: dict[str, Any]) -> dict[str, Any]:
        name = str(file_info.get("name") or "")
        if not name:
            raise SilentTextLocalizationError("Gemini file name is missing.")
        for _attempt in range(60):
            raw_state = file_info.get("state")
            state = str(
                raw_state.get("name")
                if isinstance(raw_state, dict)
                else raw_state or ""
            ).upper()
            if state == "ACTIVE":
                return file_info
            if state == "FAILED":
                raise SilentTextLocalizationError(
                    f"Gemini failed to process uploaded file {name}."
                )
            time.sleep(5)
            file_info = self._request(
                "GET",
                f"{GEMINI_API_ROOT}/v1beta/{name}",
            ).json()
        raise SilentTextLocalizationError(
            f"Gemini file processing timed out: {name}."
        )

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        media: list[dict[str, Any]],
        inline_media_paths: list[str | Path] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 65536,
        request_timeout_seconds: int | None = None,
        request_attempts: int = 3,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        parts = [
            {
                "file_data": {
                    "mime_type": item.get("mime_type"),
                    "file_uri": item.get("uri"),
                }
            }
            for item in media
        ]
        for raw_path in inline_media_paths or []:
            inline_path = Path(raw_path)
            inline_mime = (
                mimetypes.guess_type(inline_path.name)[0]
                or "application/octet-stream"
            )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": inline_mime,
                        "data": base64.b64encode(inline_path.read_bytes()).decode(
                            "ascii"
                        ),
                    }
                }
            )
        parts.append({"text": prompt})
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "maxOutputTokens": int(max_output_tokens),
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema or gemini_response_schema(),
            },
        }
        if thinking_level:
            payload["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": str(thinking_level)
            }
        response = self._request(
            "POST",
            f"{GEMINI_API_ROOT}/v1beta/models/{model}:generateContent",
            headers={"Content-Type": "application/json"},
            json=payload,
            request_timeout_seconds=(
                self.timeout_seconds
                if request_timeout_seconds is None
                else request_timeout_seconds
            ),
            request_attempts=request_attempts,
        ).json()
        texts: list[str] = []
        for candidate in response.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
        if not texts:
            feedback = response.get("promptFeedback") if isinstance(response, dict) else None
            block_reason = (
                str(feedback.get("blockReason") or "").strip()
                if isinstance(feedback, dict)
                else ""
            )
            if block_reason:
                raise GeminiPolicyBlockError(block_reason)
            raise SilentTextLocalizationError(
                f"Gemini returned no JSON text: {response}"
            )
        return _json_object("".join(texts))

    def delete_file(self, file_info: dict[str, Any]) -> None:
        name = str(file_info.get("name") or "")
        if not name:
            return
        try:
            self._request("DELETE", f"{GEMINI_API_ROOT}/v1beta/{name}")
        except Exception:
            pass


class AntigravityRestClient(GeminiRestClient):
    """Bounded Antigravity Interactions client for fail-closed Lite triage.

    This intentionally accepts only inline contact sheets.  It is not a
    replacement for the deterministic VSR stages or the strong Gemini review.
    """

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        media: list[dict[str, Any]],
        inline_media_paths: list[str | Path] | None = None,
        response_schema: dict[str, Any] | None = None,
        max_output_tokens: int = 65536,
        request_timeout_seconds: int | None = None,
        request_attempts: int = 3,
        thinking_level: str | None = None,
    ) -> dict[str, Any]:
        if media:
            raise SilentTextLocalizationError(
                "Antigravity triage accepts contact sheets only."
            )
        input_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for raw_path in inline_media_paths or []:
            inline_path = Path(raw_path)
            input_parts.append(
                {
                    "type": "image",
                    "mime_type": (
                        mimetypes.guess_type(inline_path.name)[0]
                        or "application/octet-stream"
                    ),
                    "data": base64.b64encode(inline_path.read_bytes()).decode("ascii"),
                }
            )
        payload = {
            "agent": ANTIGRAVITY_AGENT,
            "input": input_parts,
            "environment": "remote",
            "agent_config": {
                "type": "antigravity",
                "model": model or ANTIGRAVITY_TRIAGE_MODEL,
                "max_total_tokens": ANTIGRAVITY_TRIAGE_TOKEN_BUDGET,
            },
        }
        response = self._request(
            "POST",
            f"{GEMINI_API_ROOT}/v1beta/interactions",
            headers={
                "Content-Type": "application/json",
                "Api-Revision": "2026-05-20",
            },
            json=payload,
            request_timeout_seconds=(
                self.timeout_seconds
                if request_timeout_seconds is None
                else request_timeout_seconds
            ),
            request_attempts=request_attempts,
        ).json()
        status = str(response.get("status") or "completed").lower()
        if status != "completed":
            raise GeminiRequestError(
                f"Antigravity interaction did not complete ({status}).",
                rotate_key=status in {"failed", "incomplete"},
            )
        text = response.get("output_text") or response.get("outputText")
        if not isinstance(text, str) or not text.strip():
            # The REST wire format can expose completed agent output in steps
            # rather than the SDK convenience field output_text.  Extract only
            # text values; never serialize the response because it may include
            # provider metadata we do not want in Kaggle logs.
            def text_values(value: object) -> list[str]:
                if isinstance(value, dict):
                    values: list[str] = []
                    for key, child in value.items():
                        if key in {"text", "output_text", "outputText"} and isinstance(child, str):
                            values.append(child)
                        elif key in {"output", "outputs", "content", "parts", "steps", "result"}:
                            values.extend(text_values(child))
                    return values
                if isinstance(value, list):
                    return [item for child in value for item in text_values(child)]
                return []

            candidates = text_values(response)
            text = candidates[-1] if candidates else ""
        if not isinstance(text, str) or not text.strip():
            response_keys = ",".join(sorted(str(key) for key in response)[:8])
            raise SilentTextLocalizationError(
                "Antigravity returned no readable text (response fields: "
                f"{response_keys or 'none'})."
            )
        return _json_object(text)


def _review_summary(decisions: list[dict[str, Any]]) -> dict[str, int]:
    restore_count = sum(
        item.get("render_action") == "restore_original" for item in decisions
    )
    return {
        "event_count": len(decisions),
        "selected_for_inpaint_count": sum(
            bool(item.get("selected_for_inpaint")) for item in decisions
        ),
        "spoken_remove_count": sum(
            item.get("classification") == "spoken_subtitle"
            and item.get("render_action") == "remove_original"
            for item in decisions
        ),
        "auto_approved_count": sum(
            bool(item.get("auto_approved")) for item in decisions
        ),
        "replace_count": sum(
            item.get("render_action") == "replace_with_vietnamese"
            for item in decisions
        ),
        "remove_count": sum(
            item.get("render_action") == "remove_original"
            for item in decisions
        ),
        "restore_count": restore_count,
        "semantic_keep_decision_count": restore_count,
        "linked_fragment_remove_count": sum(
            item.get("routing_override") == "same_frame_text_fragment"
            for item in decisions
        ),
    }


def _gemini_evidence_scope(*, srt_text: str) -> dict[str, Any]:
    """Describe only evidence that is actually attached to Gemini requests."""

    return {
        "contact_sheets": True,
        "ocr_text": True,
        "event_timing_and_geometry": True,
        "srt": bool(srt_text.strip()),
        "continuous_video": False,
        "audio": False,
        "policy": "contact_sheets_ocr_timing_geometry_optional_srt",
    }


def _local_spoken_decision(event: dict[str, Any]) -> dict[str, Any]:
    """Create the remove-only decision for a trusted local OCR/SRT match."""
    relation = dict(event.get("speech_relation") or {})
    start_ms = int(event.get("start_ms") or 0)
    end_ms = int(event.get("end_ms") or start_ms)
    duration_seconds = float(
        event.get("duration_seconds") or max(0, end_ms - start_ms) / 1000.0
    )
    confidence = float(relation.get("combined_score") or 0.0)
    source_text = str(event.get("ocr_best_text") or "").strip()
    return {
        "track_id": str(event.get("track_id") or ""),
        "classification": "spoken_subtitle",
        "text_origin": "post_production_overlay",
        "source_text_corrected": source_text,
        "vietnamese_translation": "",
        "highlight_terms": [],
        "replacement_action": "keep_removed",
        "confidence": confidence,
        "reason_vi": (
            "OCR khớp chắc với lời thoại SRT theo cả nội dung và thời gian; "
            "xóa lớp phụ đề gốc và không render bản dịch."
        ),
        "auto_approved": False,
        "selected_for_inpaint": True,
        "render_action": "remove_original",
        "decision_source": "local_ocr_srt_alignment",
        "speech_relation": relation,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_frame": int(event.get("start_frame") or 0),
        "end_frame": int(event.get("end_frame") or 0),
        "duration_seconds": round(duration_seconds, 6),
        "eligible_for_auto_translation": bool(
            event.get(
                "eligible_for_auto_translation",
                duration_seconds + 1e-9 >= MIN_AUTO_TRANSLATION_RENDER_SECONDS,
            )
        ),
        "bbox": event.get("bbox") or [],
        "bbox_normalized": event.get("bbox_normalized") or [],
        "visual_style": event.get("visual_style") or {},
    }


def _trusted_local_spoken_events(profile: dict[str, Any]) -> list[dict[str, Any]]:
    annotate_speech_review_routes(profile)
    return [
        event
        for event in profile.get("events") or []
        if event.get("review_route") == "local_spoken_remove_only"
    ]


def _is_short_small_middle_event(
    event: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    """Return whether a weak central micro-text event should remain untouched.

    This is deliberately a conservative routing rule, not a classification
    claim.  Brief small OCR hits in the middle of a frame are commonly scene
    details or detector noise.  Text in the top and bottom bands remains in
    Gemini because those positions frequently contain editorial overlays.
    Missing or malformed geometry always falls back to Gemini review.
    """
    if not config.get("middle_microtext_filter_enabled"):
        return False
    try:
        duration = float(
            event.get("effective_duration_seconds")
            or event.get("duration_seconds")
            or 0.0
        )
        box = [float(value) for value in event.get("bbox_normalized") or []]
    except (TypeError, ValueError):
        return False
    if len(box) != 4:
        return False
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False
    center_y = (y0 + y1) / 2.0
    return bool(
        duration <= float(config["middle_microtext_max_duration_seconds"])
        and width * height <= float(config["middle_microtext_max_area_fraction"])
        and height <= float(config["middle_microtext_max_height_fraction"])
        and float(config["middle_microtext_top_fraction"])
        < center_y
        < float(config["middle_microtext_bottom_fraction"])
    )


def _local_middle_microtext_keep_decision(event: dict[str, Any]) -> dict[str, Any]:
    """Record a fail-closed keep decision for a central micro-text event."""
    start_ms = int(event.get("start_ms") or 0)
    end_ms = int(event.get("end_ms") or start_ms)
    duration_seconds = float(
        event.get("effective_duration_seconds")
        or event.get("duration_seconds")
        or max(0, end_ms - start_ms) / 1000.0
    )
    return {
        "track_id": str(event.get("track_id") or ""),
        "classification": "uncertain",
        "text_origin": "uncertain",
        "source_text_corrected": str(event.get("ocr_best_text") or "").strip(),
        "vietnamese_translation": "",
        "highlight_terms": [],
        "replacement_action": "ignore",
        "confidence": 1.0,
        "reason_vi": (
            "Chữ nhỏ, xuất hiện ngắn ở vùng giữa khung hình; giữ nguyên để "
            "tránh xoá nhầm và không gửi Gemini."
        ),
        "auto_approved": False,
        "selected_for_inpaint": False,
        "render_action": "restore_original",
        "decision_source": "local_short_small_middle_keep",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_frame": int(event.get("start_frame") or 0),
        "end_frame": int(event.get("end_frame") or 0),
        "duration_seconds": round(duration_seconds, 6),
        "eligible_for_auto_translation": False,
        "bbox": event.get("bbox") or [],
        "bbox_normalized": event.get("bbox_normalized") or [],
        "visual_style": event.get("visual_style") or {},
    }


def _box_center_in_expanded_text_band(
    candidate_box: list[int] | tuple[int, ...],
    anchor_box: list[int] | tuple[int, ...],
) -> bool:
    if len(candidate_box) != 4 or len(anchor_box) != 4:
        return False
    cx0, cx1, cy0, cy1 = (int(value) for value in candidate_box)
    ax0, ax1, ay0, ay1 = (int(value) for value in anchor_box)
    if cx1 <= cx0 or cy1 <= cy0 or ax1 <= ax0 or ay1 <= ay0:
        return False
    center_x = (cx0 + cx1) / 2
    center_y = (cy0 + cy1) / 2
    anchor_width = ax1 - ax0
    anchor_height = ay1 - ay0
    pad_x = max(4.0, anchor_width * 0.12)
    pad_y = max(4.0, anchor_height * 0.60)
    return bool(
        ax0 - pad_x <= center_x <= ax1 + pad_x
        and ay0 - pad_y <= center_y <= ay1 + pad_y
    )


def _link_short_text_fragments(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Keep one-frame OCR fragments from escaping a confirmed text track.

    OCR/Stage7 can split the first or last strokes of one subtitle into a
    separate short track.  Gemini then sees the fragment without enough glyph
    context and may call it scenery.  Exact same-frame ownership plus spatial
    containment is stronger evidence: the fragment is erased, but never gets a
    second Vietnamese render.
    """

    # Semantic OCR profiles already fan one text decision out through the
    # explicit semantic-to-exact mask graph. Re-linking short *semantic*
    # events here would let geometry override Gemini's keep decision and can
    # erase unrelated scene text/logo near an overlay. Keep this compatibility
    # heuristic only for downloaded legacy profiles whose events were exact
    # mask fragments.
    if isinstance((profile.get("mask_mapping") or {}).get("exact_tracks"), list):
        return 0

    event_by_id = {
        str(event.get("track_id") or ""): event
        for event in profile.get("events") or []
    }

    decision_by_id = {
        str(decision.get("track_id") or ""): decision
        for decision in decisions
    }
    selected_ids = {
        track_id
        for track_id, decision in decision_by_id.items()
        if bool(decision.get("selected_for_inpaint"))
        and decision.get("render_action")
        in {"remove_original", "replace_with_vietnamese"}
    }
    frame_boxes_by_id: dict[str, dict[int, list[int]]] = {}
    for track_id, event in event_by_id.items():
        frame_boxes: dict[int, list[int]] = {}
        for observation in event.get("frame_boxes") or []:
            if not isinstance(observation, dict):
                continue
            box = observation.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            frame_no = int(observation.get("frame_no") or 0)
            if frame_no > 0:
                frame_boxes[frame_no] = [int(value) for value in box]
        frame_boxes_by_id[track_id] = frame_boxes

    linked_count = 0
    for track_id, decision in decision_by_id.items():
        if bool(decision.get("selected_for_inpaint")):
            continue
        event = event_by_id.get(track_id) or {}
        if float(event.get("duration_seconds") or 0.0) >= 0.5:
            continue
        candidate_frames = frame_boxes_by_id.get(track_id) or {}
        if not candidate_frames:
            continue
        best_anchor = ""
        best_linked_frames = 0
        for anchor_id in selected_ids:
            anchor_frames = frame_boxes_by_id.get(anchor_id) or {}
            linked_frames = sum(
                _box_center_in_expanded_text_band(
                    candidate_box,
                    anchor_frames[frame_no],
                )
                for frame_no, candidate_box in candidate_frames.items()
                if frame_no in anchor_frames
            )
            if linked_frames > best_linked_frames:
                best_anchor = anchor_id
                best_linked_frames = linked_frames
        required_frames = max(1, math.ceil(len(candidate_frames) * 0.60))
        if not best_anchor or best_linked_frames < required_frames:
            continue
        decision["selected_for_inpaint"] = True
        decision["render_action"] = "remove_original"
        decision["auto_approved"] = False
        decision["routing_override"] = "same_frame_text_fragment"
        decision["linked_text_track_id"] = best_anchor
        decision["linked_frame_count"] = best_linked_frames
        decision["reason_vi"] = (
            str(decision.get("reason_vi") or "").strip()
            + " Định tuyến an toàn: mảnh OCR rất ngắn nối cùng frame/vùng với "
            f"track chữ {best_anchor}; xoá bản gốc nhưng không render chữ lần hai."
        ).strip()
        linked_count += 1
    return linked_count


def _merged_frame_coverage(
    decisions: list[dict[str, Any]],
) -> tuple[int, int, int]:
    ranges = sorted(
        (
            max(1, int(item.get("start_frame") or 1)),
            max(
                max(1, int(item.get("start_frame") or 1)),
                int(item.get("end_frame") or item.get("start_frame") or 1),
            ),
        )
        for item in decisions
    )
    if not ranges:
        return 0, 0, 0
    merged: list[list[int]] = []
    for start_frame, end_frame in ranges:
        if merged and start_frame <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end_frame)
        else:
            merged.append([start_frame, end_frame])
    covered = sum(end_frame - start_frame + 1 for start_frame, end_frame in merged)
    return covered, ranges[0][0], max(end_frame for _, end_frame in ranges)


def _persistent_exact_mask_evidence_span(
    profile: dict[str, Any],
    envelope: list[int],
    *,
    anchor_start: int,
    anchor_end: int,
    fps: float,
) -> tuple[int, int, int]:
    """Return a conservative span backed by exact-mask geometry.

    Semantic OCR tracks can fragment across cuts even when the same small edge
    disclaimer remains on screen. Exact detector boxes are stronger geometry
    evidence than OCR text. This helper only extends a persistent overlay that
    Gemini already confirmed; it never creates a new removal decision.
    """
    exact_tracks = ((profile.get("mask_mapping") or {}).get("exact_tracks") or [])
    if not isinstance(exact_tracks, list) or len(envelope) != 4:
        return anchor_start, anchor_end, 0
    evidence_frames: set[int] = set()
    for exact_track in exact_tracks:
        if not isinstance(exact_track, dict):
            continue
        for observation in exact_track.get("frame_boxes") or []:
            if not isinstance(observation, dict):
                continue
            box = [int(value) for value in observation.get("bbox") or []]
            if len(box) != 4:
                continue
            if (
                _box_covers_target(envelope, box) < 0.72
                and _review_box_affinity(box, envelope) < 0.84
            ):
                continue
            frame_no = int(observation.get("frame_no") or 0)
            if frame_no > 0:
                evidence_frames.add(frame_no)
    if not evidence_frames:
        return anchor_start, anchor_end, 0

    maximum_gap_frames = max(
        1,
        int(round(max(1.0, fps) * PERSISTENT_EXACT_EVIDENCE_MAX_GAP_MS / 1000.0)),
    )
    components: list[list[int]] = []
    for frame_no in sorted(evidence_frames):
        if components and frame_no <= components[-1][-1] + maximum_gap_frames + 1:
            components[-1].append(frame_no)
        else:
            components.append([frame_no])
    overlapping = [
        component
        for component in components
        if component[-1] >= anchor_start and component[0] <= anchor_end
    ]
    if not overlapping:
        return anchor_start, anchor_end, 0
    component = max(
        overlapping,
        key=lambda frames: (
            min(anchor_end, frames[-1]) - max(anchor_start, frames[0]) + 1,
            len(frames),
        ),
    )
    span = max(1, component[-1] - component[0] + 1)
    minimum_observed = max(8, int(round(max(1.0, fps) * 2.0)))
    if len(component) < minimum_observed or len(component) / span < 0.25:
        return anchor_start, anchor_end, 0
    return min(anchor_start, component[0]), max(anchor_end, component[-1]), len(component)


def _link_persistent_overlay_envelopes(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Attach a stable multi-line envelope to a reviewed overlay anchor.

    Small corner disclaimers are often recognized as one reliable bottom line
    plus many empty/garbled OCR tracks covering the complete text block.  The
    latter are unsafe to erase on their own, but become strong evidence when
    they repeatedly contain a high-confidence post-production anchor at the
    same screen position across several seconds.  This rule expands only the
    inpaint ownership; it never creates an additional Vietnamese render.
    """
    video = profile.get("video") or {}
    frame_width = int(video.get("width") or 0)
    frame_height = int(video.get("height") or 0)
    fps = max(1.0, float(video.get("fps") or 0.0))
    if frame_width <= 0 or frame_height <= 0:
        return 0

    anchors = [
        item
        for item in decisions
        if bool(item.get("selected_for_inpaint"))
        and bool(item.get("persistent_fixed_overlay"))
        and bool(item.get("render_anchor_for_semantic"))
        and item.get("text_origin") == "post_production_overlay"
        and item.get("render_action") == "replace_with_vietnamese"
        and float(item.get("confidence") or 0.0) >= 0.90
    ]
    linked_ids: set[str] = set()
    maximum_gap_frames = max(12, int(round(fps * 2.0)))
    minimum_span_frames = max(12, int(round(fps * 5.0)))

    for anchor in anchors:
        anchor_id = str(anchor.get("track_id") or "")
        anchor_box = [
            int(value)
            for value in (anchor.get("persistent_bbox") or anchor.get("bbox") or [])
        ]
        if len(anchor_box) != 4:
            continue
        anchor_area = max(
            1,
            (anchor_box[1] - anchor_box[0]) * (anchor_box[3] - anchor_box[2]),
        )
        anchor_start = max(1, int(anchor.get("start_frame") or 1))
        anchor_end = max(anchor_start, int(anchor.get("end_frame") or anchor_start))
        candidates: list[dict[str, Any]] = []
        for candidate in decisions:
            candidate_id = str(candidate.get("track_id") or "")
            if not candidate_id or candidate_id == anchor_id:
                continue
            if candidate.get("render_action") == "replace_with_vietnamese":
                continue
            box = [int(value) for value in candidate.get("bbox") or []]
            if len(box) != 4:
                continue
            candidate_start = max(1, int(candidate.get("start_frame") or 1))
            candidate_end = max(
                candidate_start,
                int(candidate.get("end_frame") or candidate_start),
            )
            if (
                candidate_end < anchor_start - maximum_gap_frames
                or candidate_start > anchor_end + maximum_gap_frames
            ):
                continue
            candidate_area = max(1, (box[1] - box[0]) * (box[3] - box[2]))
            area_multiple = candidate_area / anchor_area
            if not (1.8 <= area_multiple <= 20.0):
                continue
            if candidate_area / (frame_width * frame_height) > 0.03:
                continue
            near_edge = bool(
                box[2] / frame_height <= 0.28
                or box[0] / frame_width <= 0.08
                or box[1] / frame_width >= 0.92
            )
            if not near_edge or _review_box_affinity(box, anchor_box) < 0.85:
                continue
            candidates.append(candidate)

        clusters: list[list[dict[str, Any]]] = []
        for candidate in candidates:
            box = candidate.get("bbox") or []
            target = next(
                (
                    cluster
                    for cluster in clusters
                    if max(
                        _review_box_affinity(member.get("bbox"), box)
                        for member in cluster
                    )
                    >= 0.82
                ),
                None,
            )
            if target is None:
                clusters.append([candidate])
            else:
                target.append(candidate)

        qualified: list[tuple[int, list[dict[str, Any]], int, int]] = []
        for cluster in clusters:
            covered, cluster_start, cluster_end = _merged_frame_coverage(cluster)
            span = max(1, cluster_end - cluster_start + 1)
            if (
                len(cluster) >= 3
                and span >= minimum_span_frames
                and covered / span >= 0.25
            ):
                qualified.append((covered, cluster, cluster_start, cluster_end))
        if not qualified:
            continue
        _, cluster, cluster_start, cluster_end = max(
            qualified,
            key=lambda item: (item[0], len(item[1])),
        )
        boxes = [[int(value) for value in item.get("bbox") or []] for item in cluster]
        envelope = [
            min(box[0] for box in boxes),
            max(box[1] for box in boxes),
            min(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]
        envelope_start = min(anchor_start, cluster_start)
        envelope_end = max(anchor_end, cluster_end)
        evidence_start, evidence_end, evidence_frame_count = _persistent_exact_mask_evidence_span(
            profile,
            envelope,
            anchor_start=envelope_start,
            anchor_end=envelope_end,
            fps=fps,
        )
        envelope_start = min(envelope_start, evidence_start)
        envelope_end = max(envelope_end, evidence_end)
        mask_envelope = _persistent_edge_mask_bbox(envelope, video)
        start_ms_values = [
            int(item.get("start_ms") or 0)
            for item in [anchor, *cluster]
            if item.get("start_ms") is not None
        ]
        end_ms_values = [
            int(item.get("end_ms") or 0)
            for item in [anchor, *cluster]
            if item.get("end_ms") is not None
        ]
        original_render_bbox = [int(value) for value in anchor.get("bbox") or []]
        anchor["persistent_bbox"] = envelope
        anchor["persistent_mask_bbox"] = list(mask_envelope)
        anchor["original_render_bbox"] = original_render_bbox
        anchor["bbox"] = envelope
        anchor["bbox_normalized"] = [
            round(envelope[0] / frame_width, 6),
            round(envelope[2] / frame_height, 6),
            round(envelope[1] / frame_width, 6),
            round(envelope[3] / frame_height, 6),
        ]
        anchor["render_target_bbox"] = envelope
        anchor["render_target_min_size"] = [
            min(frame_width, max(1, int(math.ceil((envelope[1] - envelope[0]) * 1.05)))),
            min(frame_height, max(1, int(math.ceil((envelope[3] - envelope[2]) * 1.05)))),
        ]
        visual_style = dict(
            anchor.get("visual_style")
            if isinstance(anchor.get("visual_style"), dict)
            else {}
        )
        layout = dict(
            visual_style.get("layout")
            if isinstance(visual_style.get("layout"), dict)
            else {}
        )
        layout["text_bbox"] = envelope
        layout["max_width_px"] = envelope[1] - envelope[0]
        visual_style["layout"] = layout
        font_hint = dict(
            visual_style.get("font_hint")
            if isinstance(visual_style.get("font_hint"), dict)
            else {}
        )
        font_hint["size_px"] = max(
            int(font_hint.get("size_px") or 0),
            max(16, int(round(frame_height * 0.012))),
        )
        visual_style["font_hint"] = font_hint
        anchor["visual_style"] = visual_style
        anchor["start_frame"] = envelope_start
        anchor["end_frame"] = envelope_end
        anchor["start_ms"] = min(
            min(start_ms_values) if start_ms_values else int(
                round((envelope_start - 1) * 1000 / fps)
            ),
            int(round((envelope_start - 1) * 1000 / fps)),
        )
        anchor["end_ms"] = max(
            max(end_ms_values) if end_ms_values else int(
                round(envelope_end * 1000 / fps)
            ),
            int(round(envelope_end * 1000 / fps)),
        )
        if evidence_frame_count:
            anchor["persistent_exact_evidence_frame_count"] = evidence_frame_count
            anchor["persistent_exact_evidence_start_frame"] = evidence_start
            anchor["persistent_exact_evidence_end_frame"] = evidence_end
        anchor["duration_seconds"] = round(
            max(0, int(anchor["end_ms"]) - int(anchor["start_ms"])) / 1000.0,
            6,
        )
        anchor["persistent_envelope_track_ids"] = [
            str(item.get("track_id") or "") for item in cluster
        ]
        for candidate in cluster:
            candidate_id = str(candidate.get("track_id") or "")
            candidate["selected_for_inpaint"] = True
            candidate["render_action"] = "remove_original"
            candidate["auto_approved"] = False
            candidate["persistent_fixed_overlay"] = True
            candidate["persistent_bbox"] = envelope
            candidate["persistent_mask_bbox"] = list(mask_envelope)
            candidate["routing_override"] = "reviewed_persistent_overlay_envelope"
            candidate["linked_text_track_id"] = anchor_id
            candidate["reason_vi"] = (
                str(candidate.get("reason_vi") or "").strip()
                + " Định tuyến an toàn: bbox nhiều dòng cố định bao chứa một "
                f"overlay hậu kỳ đã được xác nhận ({anchor_id}); chỉ mở rộng vùng "
                "xóa, không render thêm bản dịch."
            ).strip()
            linked_ids.add(candidate_id)
    return len(linked_ids)


def _request_scenery_guard(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    clients: list[GeminiRestClient],
    log: Callable[[str], None],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    candidates = _scenery_guard_candidates(profile)
    if not candidates:
        return {}, []
    eligible_count = len(candidates)
    candidates.sort(
        key=lambda event: (
            set(event.get("detection_sources") or []) != {"stage7"},
            int(event.get("observed_frames") or 0),
            float(event.get("ocr_best_confidence") or 0.0),
            str(event.get("track_id") or ""),
        )
    )
    if eligible_count > SCENERY_GUARD_MAX_CANDIDATES:
        candidates = candidates[:SCENERY_GUARD_MAX_CANDIDATES]
        log(
            "Gemini Flash Lite scenery guard limited to the "
            f"{len(candidates)} highest-risk OCR candidates out of "
            f"{eligible_count}; the strong Gemini review will still inspect "
            "every track."
        )
    batch_count = math.ceil(len(candidates) / CONTACT_SHEET_EVENTS)
    log(
        "Gemini Flash Lite scenery guard started: "
        f"candidates={len(candidates)}; batches={batch_count}."
    )
    results: dict[str, dict[str, Any]] = {}
    used_models: list[str] = []
    profile_source = Path(profile_path)
    for start in range(0, len(candidates), CONTACT_SHEET_EVENTS):
        batch = candidates[start : start + CONTACT_SHEET_EVENTS]
        batch_result = None
        for key_index, gemini in enumerate(clients, start=1):
            uploads: list[dict[str, Any]] = []
            try:
                for event in batch:
                    image_path = _safe_asset(
                        profile_source,
                        event.get("representative_frame"),
                    )
                    uploads.append(gemini.upload_file(image_path))
                for attempt in range(2):
                    response = gemini.generate(
                        model=SCENERY_GUARD_MODEL,
                        prompt=build_scenery_guard_prompt(batch),
                        media=uploads,
                        response_schema=scenery_guard_response_schema(),
                        max_output_tokens=4096,
                        request_timeout_seconds=60,
                        request_attempts=1,
                    )
                    try:
                        batch_result = validate_scenery_guard_response(
                            batch,
                            response,
                        )
                        break
                    except SilentTextLocalizationError:
                        if attempt:
                            raise
                        log(
                            "Gemini Flash Lite scenery guard returned an invalid "
                            "contract; retrying once."
                        )
                if batch_result is not None:
                    used_models.append(SCENERY_GUARD_MODEL)
                    break
            except (GeminiRequestError, SilentTextLocalizationError) as exc:
                log(
                    "Gemini Flash Lite scenery guard was unavailable on key "
                    f"{key_index}/{len(clients)}; trying the next key."
                )
                if isinstance(exc, GeminiRequestError) and not exc.rotate_key:
                    break
            finally:
                for uploaded in uploads:
                    gemini.delete_file(uploaded)
        if batch_result is None:
            # Guard failure is fail-open into the stronger 3.6 review.  Lite is
            # never allowed to decide text handling when it is uncertain.
            batch_result = {
                str(event.get("track_id") or ""): {
                    "track_id": str(event.get("track_id") or ""),
                    "verdict": "UNCERTAIN",
                    "confidence": 0.0,
                    "reason_vi": "Bộ lọc cảnh vật không khả dụng; chuyển Gemini 3.6.",
                }
                for event in batch
            }
        results.update(batch_result)
        log(
            "Gemini Flash Lite scenery guard progress: "
            f"{min(start + len(batch), len(candidates))}/{len(candidates)}."
        )
    log(
        "Gemini Flash Lite scenery guard completed: "
        f"reviewed={len(results)}."
    )
    return results, used_models


def _normalized_review_text(value: object) -> str:
    return "".join(
        character.lower()
        for character in str(value or "").strip()
        if character.isalnum()
    )


def _review_texts_match(first: object, second: object) -> bool:
    left = _normalized_review_text(first)
    right = _normalized_review_text(second)
    if not left or not right:
        return False
    if left == right:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= 3 and shorter in longer and len(shorter) / len(longer) >= 0.72:
        return True
    return SequenceMatcher(None, left, right).ratio() >= 0.94


def _focused_replacement_has_ocr_support(
    event: dict[str, Any],
    decision: dict[str, Any],
) -> bool:
    """Require local OCR evidence before an uncertain track becomes a replacement."""
    corrected = str(decision.get("source_text_corrected") or "").strip()
    if not corrected:
        return False
    candidates = [
        item
        for item in event.get("ocr_candidates") or []
        if isinstance(item, dict)
    ]
    best_text = str(event.get("ocr_best_text") or "").strip()
    if best_text:
        candidates.append(
            {
                "text": best_text,
                "confidence": event.get("ocr_best_confidence", 0.0),
            }
        )
    for candidate in candidates:
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE:
            continue
        if _review_texts_match(candidate.get("text"), corrected):
            return True
    return False


def _focused_second_review_has_local_ocr_evidence(event: dict[str, Any]) -> bool:
    """Return whether an uncertain event has OCR evidence worth a second review.

    Editorial/UI correction candidates are handled separately and are never
    suppressed by this gate. This only prevents a fail-closed uncertain event
    with no sufficiently reliable local OCR text from spending a Gemini call
    that cannot later pass the replacement evidence guard anyway.
    """
    candidates = [
        item
        for item in event.get("ocr_candidates") or []
        if isinstance(item, dict)
    ]
    best_text = str(event.get("ocr_best_text") or "").strip()
    if best_text:
        candidates.append(
            {
                "text": best_text,
                "confidence": event.get("ocr_best_confidence", 0.0),
            }
        )
    for candidate in candidates:
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE:
            continue
        normalized = _normalized_review_text(candidate.get("text"))
        if len(normalized) >= FOCUSED_REVIEW_OCR_SUPPORT_MIN_TEXT_LENGTH:
            return True
        if len(normalized) != 1:
            continue
        if normalized.isascii():
            continue
        try:
            duration_seconds = float(event.get("duration_seconds") or 0.0)
        except (TypeError, ValueError):
            duration_seconds = 0.0
        if duration_seconds <= 0.0:
            try:
                duration_seconds = max(
                    0.0,
                    (float(event.get("end_ms") or 0.0) - float(event.get("start_ms") or 0.0))
                    / 1000.0,
                )
            except (TypeError, ValueError):
                duration_seconds = 0.0
        if duration_seconds >= MIDDLE_MICROTEXT_MAX_DURATION_SECONDS:
            return True
        bbox = event.get("bbox_normalized")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            try:
                x1, y1, x2, y2 = (float(value) for value in bbox)
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
            except (TypeError, ValueError):
                pass
            else:
                if (
                    width * height >= MIDDLE_MICROTEXT_MAX_AREA_FRACTION
                    or height >= MIDDLE_MICROTEXT_MAX_HEIGHT_FRACTION
                ):
                    return True
    return False


def _focused_second_review_upgrade_allowed(
    primary: dict[str, Any],
    focused: dict[str, Any],
    event: dict[str, Any],
) -> bool:
    """Fail closed when a second review invents a replacement for weak OCR."""
    if focused.get("replacement_action") != "replace_with_vietnamese":
        return True
    primary_was_fail_closed = bool(
        primary.get("classification") == "uncertain"
        or primary.get("text_origin") == "uncertain"
        or primary.get("replacement_action") == "needs_review"
    )
    if not primary_was_fail_closed:
        return True
    return _focused_replacement_has_ocr_support(event, focused)


def _editorial_ui_second_review_ids(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> set[str]:
    """Find likely sentence-like editorial overlays mislabeled as branding/UI.

    This is only a trigger for a focused Gemini review, never a local origin
    override.  Scene text and real UI remain preserved if the focused review
    confirms them.
    """
    event_by_id = {
        str(event.get("track_id") or ""): event
        for event in profile.get("events") or []
    }
    result: set[str] = set()
    for decision in decisions:
        if decision.get("replacement_action") != "ignore":
            continue
        if (
            decision.get("classification") != "decorative_overlay"
            and decision.get("text_origin") != "ui_or_watermark"
        ):
            continue
        track_id = str(decision.get("track_id") or "")
        event = event_by_id.get(track_id) or {}
        normalized = _normalized_review_text(
            event.get("ocr_best_text")
            or decision.get("source_text_corrected")
        )
        if len(normalized) < 8:
            continue
        bbox = event.get("bbox_normalized")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        width = x2 - x1
        height = y2 - y1
        if width < 0.22 or height <= 0 or height > 0.35:
            continue
        result.add(track_id)
    return result


def _is_weak_primary_review_ocr_fragment(event: dict[str, Any]) -> bool:
    """Route tiny blank detector fragments around the expensive strong review.

    Quality-recall tiles deliberately increase detector sensitivity. A few-frame
    box with no recognized glyphs is useful mask evidence, but it is not a
    meaningful semantic object by itself. Keep it fail-closed locally so later
    visual/persistent ownership can still attach its exact mask to a reviewed
    anchor without spending one strong-review slot per detector fragment.
    """
    candidates = [
        item
        for item in event.get("ocr_candidates") or []
        if isinstance(item, dict)
    ]
    best_text = _normalized_review_text(event.get("ocr_best_text"))
    if best_text or any(_normalized_review_text(item.get("text")) for item in candidates):
        return False
    if _focused_second_review_has_local_ocr_evidence(event):
        return False
    try:
        duration_seconds = float(event.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        return False
    try:
        observed_frames = int(event.get("observed_frames") or 0)
    except (TypeError, ValueError):
        observed_frames = 0
    if observed_frames <= 0:
        observed_frames = len(
            [item for item in event.get("frame_boxes") or [] if isinstance(item, dict)]
        )
    return bool(
        0.0 <= duration_seconds <= PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_DURATION_SECONDS
        and observed_frames <= PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_OBSERVED_FRAMES
    )


def _local_weak_ocr_fragment_keep_decision(event: dict[str, Any]) -> dict[str, Any]:
    """Keep a blank few-frame detector fragment until visual ownership resolves it."""
    start_ms = int(event.get("start_ms") or 0)
    end_ms = int(event.get("end_ms") or start_ms)
    duration_seconds = float(
        event.get("effective_duration_seconds")
        or event.get("duration_seconds")
        or max(0, end_ms - start_ms) / 1000.0
    )
    return {
        "track_id": str(event.get("track_id") or ""),
        "classification": "uncertain",
        "text_origin": "uncertain",
        "source_text_corrected": "",
        "vietnamese_translation": "",
        "highlight_terms": [],
        "design_role": "neutral",
        "replacement_action": "needs_review",
        "confidence": 1.0,
        "reason_vi": (
            "Mảnh detector rất ngắn không có glyph OCR đáng tin; giữ nguyên ở "
            "lượt semantic chính và để ownership hình học/persistent quyết định mask."
        ),
        "auto_approved": False,
        "selected_for_inpaint": False,
        "render_action": "restore_original",
        "decision_source": "local_weak_ocr_fragment_gate",
        "routing_override": "weak_ocr_fragment_deferred_to_visual_owner",
        "start_ms": start_ms,
        "end_ms": end_ms,
        "start_frame": int(event.get("start_frame") or 0),
        "end_frame": int(event.get("end_frame") or 0),
        "duration_seconds": round(duration_seconds, 6),
        "eligible_for_auto_translation": False,
        "bbox": event.get("bbox") or [],
        "bbox_normalized": event.get("bbox_normalized") or [],
        "visual_style": event.get("visual_style") or {},
    }


def _review_box_affinity(first: object, second: object) -> float:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return 0.0
    if len(first) != 4 or len(second) != 4:
        return 0.0
    ax0, ax1, ay0, ay1 = (float(value) for value in first)
    bx0, bx1, by0, by1 = (float(value) for value in second)
    aw = max(1.0, ax1 - ax0)
    ah = max(1.0, ay1 - ay0)
    bw = max(1.0, bx1 - bx0)
    bh = max(1.0, by1 - by0)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    overlap = intersection / max(1.0, min(aw * ah, bw * bh))
    acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    horizontal_score = max(
        0.0,
        1.0 - abs(acx - bcx) / max(aw, bw, 1.0),
    )
    vertical_score = max(
        0.0,
        1.0 - abs(acy - bcy) / max(ah, bh, 1.0),
    )
    center_score = min(horizontal_score, vertical_score)
    return max(overlap, center_score * 0.72)


def _high_confidence_review_texts(event: dict[str, Any]) -> list[str]:
    """Return normalized OCR hypotheses suitable for semantic identity only."""
    values: list[str] = []
    best = _normalized_review_text(event.get("ocr_best_text"))
    try:
        best_confidence = float(event.get("ocr_best_confidence") or 0.0)
    except (TypeError, ValueError):
        best_confidence = 0.0
    if best and best_confidence >= FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE:
        values.append(best)
    for candidate in event.get("ocr_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if confidence < FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE:
            continue
        text = _normalized_review_text(candidate.get("text"))
        if text and text not in values:
            values.append(text)
    return values


def _semantic_identity_text(event: dict[str, Any]) -> str:
    """Choose a stable OCR identity without trusting top-1 partial reads blindly.

    Recognition over several representative crops can rank a truncated string a
    few thousandths above the full string. For grouping only, prefer a longer
    close-confidence hypothesis when it is textually compatible with the best
    read. Unrelated candidates are never allowed to replace the top hypothesis.
    """
    hypotheses: list[tuple[str, float]] = []
    best_text = _normalized_review_text(event.get("ocr_best_text"))
    try:
        best_confidence = float(event.get("ocr_best_confidence") or 0.0)
    except (TypeError, ValueError):
        best_confidence = 0.0
    if best_text:
        hypotheses.append((best_text, best_confidence))
    for candidate in event.get("ocr_candidates") or []:
        if not isinstance(candidate, dict):
            continue
        text = _normalized_review_text(candidate.get("text"))
        if not text:
            continue
        try:
            confidence = float(candidate.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        hypotheses.append((text, confidence))
    if not hypotheses:
        return ""
    peak = max(confidence for _text, confidence in hypotheses)
    floor = max(FOCUSED_REVIEW_OCR_SUPPORT_MIN_CONFIDENCE, peak - 0.08)
    base_text, _base_confidence = max(
        hypotheses,
        key=lambda item: (item[1], item[0] == best_text, len(item[0])),
    )
    eligible = []
    for text, confidence in hypotheses:
        if confidence < floor:
            continue
        shorter, longer = sorted((base_text, text), key=len)
        compatible = bool(
            text == base_text
            or (shorter and shorter in longer)
            or SequenceMatcher(None, base_text, text).ratio() >= 0.72
        )
        if compatible:
            eligible.append((text, confidence))
    if not eligible:
        return base_text
    return max(eligible, key=lambda item: (len(item[0]), item[1], item[0]))[0]


def _stable_repeated_text_owner_match(
    first: dict[str, Any],
    second: dict[str, Any],
    affinity: float,
) -> bool:
    """Recognize the same reviewed text object reappearing at one screen slot.

    Repeated name cards and editor labels often disappear across scene cuts.
    Their semantics should be reviewed once, not independently per occurrence.
    Compare all high-confidence OCR hypotheses, not only the top-1 hypothesis:
    tiny text can have a slightly higher-confidence partial read while another
    candidate contains the stable full label. Geometry remains the second
    independent constraint, so common text at a different screen object is not
    collapsed merely because its words match.
    """
    if affinity < 0.80:
        return False
    first_texts = _high_confidence_review_texts(first)
    second_texts = _high_confidence_review_texts(second)
    text_match = False
    for left in first_texts:
        for right in second_texts:
            if min(len(left), len(right)) < 3:
                continue
            if left == right:
                text_match = True
                break
            shorter, longer = sorted((left, right), key=len)
            if (
                len(shorter) >= 4
                and shorter in longer
                and len(shorter) / len(longer) >= 0.85
            ):
                text_match = True
                break
        if text_match:
            break
    if not text_match:
        return False
    first_box = first.get("bbox") or []
    second_box = second.get("bbox") or []
    if len(first_box) != 4 or len(second_box) != 4:
        return False
    first_area = max(
        1.0,
        (float(first_box[1]) - float(first_box[0]))
        * (float(first_box[3]) - float(first_box[2])),
    )
    second_area = max(
        1.0,
        (float(second_box[1]) - float(second_box[0]))
        * (float(second_box[3]) - float(second_box[2])),
    )
    return min(first_area, second_area) / max(first_area, second_area) >= 0.45


def _review_shared_frames(first: dict[str, Any], second: dict[str, Any]) -> int:
    first_frames = {
        int(item.get("frame_no") or 0)
        for item in first.get("frame_boxes") or []
        if isinstance(item, dict) and int(item.get("frame_no") or 0) > 0
    }
    second_frames = {
        int(item.get("frame_no") or 0)
        for item in second.get("frame_boxes") or []
        if isinstance(item, dict) and int(item.get("frame_no") or 0) > 0
    }
    if not first_frames or not second_frames:
        return 0
    return len(first_frames & second_frames)


def _edge_microtext_box(box: object, video: dict[str, Any]) -> bool:
    """Return True only for small text boxes living close to a frame edge."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        width = float(video.get("width") or 0)
        height = float(video.get("height") or 0)
        left, right, top, bottom = (float(value) for value in box)
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0 or right <= left or bottom <= top:
        return False
    width_ratio = (right - left) / width
    height_ratio = (bottom - top) / height
    area_ratio = width_ratio * height_ratio
    near_edge = bool(
        top / height <= 0.20
        or bottom / height >= 0.80
        or left / width <= 0.12
        or right / width >= 0.88
    )
    return bool(
        near_edge
        and height_ratio <= 0.055
        and width_ratio <= 0.35
        and area_ratio <= 0.012
    )


def _persistent_edge_mask_bbox(box: object, video: dict[str, Any]) -> list[int]:
    """Pad only confirmed edge-microtext masks, not their render layout."""
    if not _edge_microtext_box(box, video):
        return [int(value) for value in box] if isinstance(box, (list, tuple)) and len(box) == 4 else []
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    left, right, top, bottom = (int(value) for value in box)
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    pad_x = max(4, int(round(box_width * 0.25)))
    pad_y = max(3, int(round(box_height * 0.12)))
    return [
        max(0, left - pad_x),
        min(width, right + pad_x),
        max(0, top - pad_y),
        min(height, bottom + pad_y),
    ]


def _persistent_edge_render_bbox(box: object, video: dict[str, Any]) -> list[int]:
    """Create a wider, shallow render-only box for confirmed edge microtext."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    width = int(video.get("width") or 0)
    height = int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        return []
    left, right, top, bottom = (int(value) for value in box)
    if right <= left or bottom <= top:
        return []
    box_width = max(1, right - left)
    box_height = max(1, bottom - top)
    safe_x = max(8, int(round(width * 0.025)))
    safe_y = max(6, int(round(height * 0.015)))
    target_width = min(
        width - safe_x * 2,
        max(int(round(box_width * 1.90)), int(round(width * 0.24))),
        int(round(width * 0.32)),
    )
    target_height = min(
        height - safe_y * 2,
        max(int(round(box_height * 1.15)), int(round(height * 0.045))),
        int(round(height * 0.10)),
    )
    anchor_right = min(width - safe_x, right + int(round(width * 0.02)))
    new_left = max(safe_x, anchor_right - target_width)
    new_right = min(width - safe_x, new_left + target_width)
    new_top = max(safe_y, top - max(2, int(round(box_height * 0.05))))
    new_bottom = min(height - safe_y, new_top + target_height)
    if new_bottom - new_top < target_height:
        new_top = max(safe_y, new_bottom - target_height)
    return [int(new_left), int(new_right), int(new_top), int(new_bottom)]


def _persistent_edge_disclaimer(decision: dict[str, Any]) -> bool:
    source = str(decision.get("source_text_corrected") or "")
    translated = str(decision.get("vietnamese_translation") or "").lower()
    return bool(
        "请勿模仿" in source
        or "剧情纯属虚构" in source
        or ("bắt chước" in translated and "hư cấu" in translated)
    )


def _persistent_edge_candidate(event: dict[str, Any], video: dict[str, Any]) -> bool:
    """Identify small, stationary edge overlays without guessing their meaning."""
    text = _normalized_review_text(event.get("ocr_best_text"))
    return bool(
        len(text) >= 4
        and _edge_microtext_box(event.get("bbox"), video)
        and int(event.get("observed_frames") or 0) >= 2
    )


def _persistent_fragment_text_compatible(anchor_text: object, fragment_text: object) -> bool:
    anchor = _normalized_review_text(anchor_text)
    fragment = _normalized_review_text(fragment_text)
    if not fragment:
        return True
    if not anchor:
        return False
    if _review_texts_match(anchor, fragment):
        return True
    if fragment in anchor or anchor in fragment:
        return True
    if len(fragment) <= 3 and all(char in anchor for char in fragment):
        return True
    return False


def _promote_reviewed_persistent_edge_groups(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Promote stable reviewed edge microtext after Gemini corrected its text."""
    video = dict(profile.get("video") or {})
    candidates = [
        item
        for item in decisions
        if bool(item.get("selected_for_inpaint"))
        and item.get("text_origin") == "post_production_overlay"
        and float(item.get("confidence") or 0.0) >= 0.90
        and len(_normalized_review_text(item.get("source_text_corrected"))) >= 4
        and _edge_microtext_box(item.get("bbox"), video)
    ]
    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        target = next(
            (
                group
                for group in groups
                if _review_texts_match(
                    group[0].get("source_text_corrected"),
                    candidate.get("source_text_corrected"),
                )
                and max(
                    _review_box_affinity(member.get("bbox"), candidate.get("bbox"))
                    for member in group
                ) >= 0.72
            ),
            None,
        )
        if target is None:
            groups.append([candidate])
        else:
            target.append(candidate)

    promoted_group_count = 0
    for group in groups:
        if len(group) < 3:
            continue
        ranges = sorted(
            (
                int(item.get("start_ms") or 0),
                int(item.get("end_ms") or item.get("start_ms") or 0),
            )
            for item in group
        )
        span_ms = max(1, ranges[-1][1] - ranges[0][0])
        if span_ms < 5_000:
            continue
        merged: list[list[int]] = []
        for start_ms, end_ms in ranges:
            if merged and start_ms <= merged[-1][1] + 500:
                merged[-1][1] = max(merged[-1][1], end_ms)
            else:
                merged.append([start_ms, end_ms])
        covered_ms = sum(max(0, end - start) for start, end in merged)
        if covered_ms / span_ms < 0.15:
            continue
        boxes = [list(item.get("bbox") or []) for item in group]
        envelope = [
            min(int(box[0]) for box in boxes),
            max(int(box[1]) for box in boxes),
            min(int(box[2]) for box in boxes),
            max(int(box[3]) for box in boxes),
        ]
        if not _edge_microtext_box(envelope, video):
            continue
        render_candidates = [
            item
            for item in group
            if item.get("render_action") == "replace_with_vietnamese"
        ]
        if not render_candidates:
            continue
        anchor = max(
            render_candidates,
            key=lambda item: (
                int(item.get("end_ms") or 0) - int(item.get("start_ms") or 0),
                int(item.get("observed_frames") or 0),
                float(item.get("confidence") or 0.0),
            ),
        )
        start_ms = min(start for start, _ in ranges)
        end_ms = max(end for _, end in ranges)
        fps = max(1.0, float(video.get("fps") or 0.0))
        start_frame = max(
            1,
            min(int(item.get("start_frame") or 1) for item in group),
        )
        end_frame = max(
            start_frame,
            max(int(item.get("end_frame") or item.get("start_frame") or 1) for item in group),
        )
        evidence_start, evidence_end, evidence_frame_count = _persistent_exact_mask_evidence_span(
            profile,
            envelope,
            anchor_start=start_frame,
            anchor_end=end_frame,
            fps=fps,
        )
        start_frame = min(start_frame, evidence_start)
        end_frame = max(end_frame, evidence_end)
        start_ms = min(start_ms, int(round((start_frame - 1) * 1000 / fps)))
        end_ms = max(end_ms, int(round(end_frame * 1000 / fps)))
        mask_envelope = _persistent_edge_mask_bbox(envelope, video)
        render_envelope = (
            _persistent_edge_render_bbox(mask_envelope, video)
            if _persistent_edge_disclaimer(anchor)
            else []
        )
        for item in group:
            item["persistent_fixed_overlay"] = True
            item["persistent_bbox"] = list(envelope)
            item["persistent_mask_bbox"] = list(mask_envelope)
            item["persistent_group_anchor_id"] = str(anchor.get("track_id") or "")
        anchor["render_anchor_for_semantic"] = True
        anchor["persistent_bbox"] = list(envelope)
        anchor["persistent_mask_bbox"] = list(mask_envelope)
        if render_envelope:
            anchor["persistent_render_bbox"] = list(render_envelope)
        anchor["start_ms"] = start_ms
        anchor["end_ms"] = end_ms
        anchor["start_frame"] = start_frame
        anchor["end_frame"] = end_frame
        if evidence_frame_count:
            anchor["persistent_exact_evidence_frame_count"] = evidence_frame_count
            anchor["persistent_exact_evidence_start_frame"] = evidence_start
            anchor["persistent_exact_evidence_end_frame"] = evidence_end
        anchor["duration_seconds"] = round(max(0, end_ms - start_ms) / 1000.0, 6)
        anchor["routing_override"] = "reviewed_persistent_edge_group_anchor"
        promoted_group_count += 1
    return promoted_group_count


def _inherit_persistent_edge_fragments(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Let tiny edge fragments inherit a confirmed persistent editor decision.

    This is intentionally narrow: only small boxes near an edge, overlapping a
    high-confidence persistent post-production anchor in time and space, and
    carrying empty/matching fragment text are promoted.  It never changes
    unrelated centre-screen or large text decisions.
    """
    video = dict(profile.get("video") or {})
    fps = max(1.0, float(video.get("fps") or 0.0))
    anchors = [
        item
        for item in decisions
        if bool(item.get("selected_for_inpaint"))
        and bool(item.get("persistent_fixed_overlay"))
        and bool(item.get("render_anchor_for_semantic"))
        and item.get("text_origin") == "post_production_overlay"
        and item.get("render_action") == "replace_with_vietnamese"
        and float(item.get("confidence") or 0.0) >= 0.90
        and _edge_microtext_box(
            item.get("persistent_bbox") or item.get("bbox"),
            video,
        )
    ]
    inherited_ids: set[str] = set()
    maximum_gap_frames = max(3, int(round(fps * 0.5)))
    for anchor in anchors:
        anchor_id = str(anchor.get("track_id") or "")
        anchor_box = list(anchor.get("persistent_bbox") or anchor.get("bbox") or [])
        anchor_text = anchor.get("source_text_corrected") or anchor.get("ocr_best_text")
        anchor_start = max(1, int(anchor.get("start_frame") or 1))
        anchor_end = max(anchor_start, int(anchor.get("end_frame") or anchor_start))
        for candidate in decisions:
            candidate_id = str(candidate.get("track_id") or "")
            if not candidate_id or candidate_id == anchor_id or candidate_id in inherited_ids:
                continue
            if bool(candidate.get("selected_for_inpaint")):
                continue
            if candidate.get("render_action") != "restore_original":
                continue
            box = candidate.get("bbox") or []
            if not _edge_microtext_box(box, video):
                continue
            candidate_start = max(1, int(candidate.get("start_frame") or 1))
            candidate_end = max(
                candidate_start,
                int(candidate.get("end_frame") or candidate_start),
            )
            if (
                candidate_end < anchor_start - maximum_gap_frames
                or candidate_start > anchor_end + maximum_gap_frames
            ):
                continue
            if _review_box_affinity(box, anchor_box) < 0.84:
                continue
            candidate_text = (
                candidate.get("source_text_corrected")
                or candidate.get("ocr_best_text")
            )
            if not _persistent_fragment_text_compatible(anchor_text, candidate_text):
                continue
            # Empty OCR boxes are useful as geometry evidence, but a small
            # nested line must not claim a much larger multi-line fragment.
            # Require directional coverage so the fragment is inherited by
            # the persistent envelope that actually owns most of its pixels.
            if (
                not _normalized_review_text(candidate_text)
                and _box_covers_target(anchor_box, list(box)) < 0.65
            ):
                continue
            candidate["selected_for_inpaint"] = True
            candidate["render_action"] = "remove_original"
            candidate["auto_approved"] = False
            candidate["persistent_fixed_overlay"] = True
            candidate["persistent_bbox"] = list(anchor_box)
            candidate["persistent_mask_bbox"] = list(
                anchor.get("persistent_mask_bbox") or anchor_box
            )
            candidate["routing_override"] = "persistent_edge_fragment_inherit"
            candidate["linked_text_track_id"] = anchor_id
            candidate.pop("mapping_guard", None)
            candidate["reason_vi"] = (
                str(candidate.get("reason_vi") or "").strip()
                + " Kế thừa cụm persistent: fragment chữ nhỏ sát biên trùng vị trí, "
                f"thời gian và nội dung với overlay hậu kỳ đã xác nhận {anchor_id}; "
                "chỉ dùng mask để xoá, không render thêm bản dịch."
            ).strip()
            inherited_ids.add(candidate_id)
    return len(inherited_ids)


def _suppress_persistent_edge_nested_renders(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Suppress short conflicting renders inside a confirmed edge overlay.

    A persistent corner disclaimer can occasionally be OCR'd as a different,
    short phrase for a few frames.  Gemini may then request a second Vietnamese
    render at the exact same place.  Suppression is allowed only when a real
    member of the confirmed persistent group is observed at the same time and
    the candidate is spatially contained by that group's small edge envelope.
    """
    video = dict(profile.get("video") or {})
    anchors = [
        item
        for item in decisions
        if item.get("routing_override") == "reviewed_persistent_edge_group_anchor"
        and bool(item.get("selected_for_inpaint"))
        and bool(item.get("persistent_fixed_overlay"))
        and bool(item.get("render_anchor_for_semantic"))
        and item.get("text_origin") == "post_production_overlay"
        and item.get("render_action") == "replace_with_vietnamese"
        and _edge_microtext_box(item.get("persistent_bbox") or item.get("bbox"), video)
    ]
    if not anchors:
        return 0
    suppressed = 0
    for candidate in decisions:
        candidate_id = str(candidate.get("track_id") or "")
        if not candidate_id:
            continue
        if not (
            bool(candidate.get("selected_for_inpaint"))
            and candidate.get("render_action") == "replace_with_vietnamese"
            and _edge_microtext_box(candidate.get("bbox"), video)
        ):
            continue
        if candidate.get("routing_override") == "reviewed_persistent_edge_group_anchor":
            continue
        candidate_group_anchor = str(candidate.get("persistent_group_anchor_id") or "")
        candidate_box = [int(value) for value in candidate.get("bbox") or []]
        candidate_start = int(candidate.get("start_ms") or 0)
        candidate_end = int(candidate.get("end_ms") or candidate_start)
        candidate_text = _normalized_review_text(candidate.get("source_text_corrected"))
        for anchor in anchors:
            anchor_id = str(anchor.get("track_id") or "")
            if not anchor_id or candidate_group_anchor not in {"", anchor_id}:
                continue
            anchor_box = [int(value) for value in anchor.get("persistent_bbox") or []]
            if len(anchor_box) != 4 or _box_covers_target(anchor_box, candidate_box) < 0.88:
                continue
            members = [
                item
                for item in decisions
                if str(item.get("persistent_group_anchor_id") or "") == anchor_id
                and str(item.get("track_id") or "") != anchor_id
            ]
            observed_now = False
            for member in members:
                member_start = int(member.get("start_ms") or 0)
                member_end = int(member.get("end_ms") or member_start)
                gap_ms = max(
                    0,
                    candidate_start - member_end,
                    member_start - candidate_end,
                )
                if gap_ms <= 500 and _review_box_affinity(
                    member.get("bbox"), candidate_box
                ) >= 0.62:
                    observed_now = True
                    break
            if not observed_now:
                continue
            anchor_text = _normalized_review_text(anchor.get("source_text_corrected"))
            compatible = _persistent_fragment_text_compatible(anchor_text, candidate_text)
            short_conflict = bool(
                candidate_text
                and anchor_text
                and len(candidate_text) <= 4
                and len(candidate_text) < len(anchor_text)
            )
            if not (compatible or short_conflict):
                continue
            candidate["render_action"] = "remove_original"
            candidate["auto_approved"] = False
            candidate["persistent_fixed_overlay"] = True
            candidate["persistent_bbox"] = list(anchor_box)
            candidate["persistent_mask_bbox"] = list(
                anchor.get("persistent_mask_bbox") or anchor_box
            )
            candidate["routing_override"] = "persistent_edge_nested_render_suppressed"
            candidate["linked_text_track_id"] = anchor_id
            candidate["reason_vi"] = (
                str(candidate.get("reason_vi") or "").strip()
                + " Chặn render lạc trong cụm persistent: cùng vùng microtext sát biên "
                f"và trùng thời gian với overlay hậu kỳ đã xác nhận {anchor_id}; "
                "giữ mask xoá nhưng không render bản dịch thứ hai."
            ).strip()
            suppressed += 1
            break
    return suppressed


def _persistent_edge_visual_superclusters(
    events: list[dict[str, Any]],
    video: dict[str, Any],
) -> tuple[list[list[dict[str, Any]]], dict[str, int]]:
    """Find one long visual track behind fragmented stationary edge OCR.

    Unlike the older persistent grouping, this deliberately does not use OCR
    text as the grouping key.  Tiny post-production disclaimers often produce
    blank, partial, or completely wrong recognition while their geometry stays
    fixed.  A supercluster is accepted only when the spatial component is
    small/edge-bound, temporally near-continuous, long-lived, and still has
    repeated-text evidence somewhere in the component.  Semantic meaning is
    left to Gemini; this function only says these observations look like one
    persistent visual object.
    """
    candidates = [
        event for event in events
        if _edge_microtext_box(event.get("bbox"), video)
    ]
    if len(candidates) < 3:
        return [], {}

    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first in enumerate(candidates):
        for second_index in range(first_index + 1, len(candidates)):
            second = candidates[second_index]
            if _review_box_affinity(first.get("bbox"), second.get("bbox")) >= 0.72:
                union(first_index, second_index)

    components: dict[int, list[dict[str, Any]]] = {}
    for index, event in enumerate(candidates):
        components.setdefault(find(index), []).append(event)

    accepted: list[list[dict[str, Any]]] = []
    cluster_by_track: dict[str, int] = {}
    for spatial_component in components.values():
        # One screen corner can host a short unrelated item before/after the
        # long-lived disclaimer. Split the spatial component at detector gaps
        # instead of rejecting the whole component because of that outlier.
        # Each temporal component must independently satisfy all persistence
        # and repeated-text evidence below.
        ordered_component = sorted(
            spatial_component,
            key=lambda event: (
                int(event.get("start_ms") or 0),
                int(event.get("end_ms") or 0),
                str(event.get("track_id") or ""),
            ),
        )
        temporal_components: list[list[dict[str, Any]]] = []
        temporal_end_ms: list[int] = []
        for event in ordered_component:
            start_ms = int(event.get("start_ms") or 0)
            end_ms = int(event.get("end_ms") or start_ms)
            if (
                temporal_components
                and start_ms <= temporal_end_ms[-1] + PERSISTENT_VISUAL_MAX_GAP_MS
            ):
                temporal_components[-1].append(event)
                temporal_end_ms[-1] = max(temporal_end_ms[-1], end_ms)
            else:
                temporal_components.append([event])
                temporal_end_ms.append(end_ms)

        for component in temporal_components:
            if len(component) < 8:
                continue
            ranges = sorted(
                (
                    int(event.get("start_ms") or 0),
                    int(event.get("end_ms") or event.get("start_ms") or 0),
                )
                for event in component
            )
            span_ms = max(1, ranges[-1][1] - ranges[0][0])
            if span_ms < 8_000:
                continue

            boxes = [list(event.get("bbox") or []) for event in component]
            envelope = [
                min(int(box[0]) for box in boxes),
                max(int(box[1]) for box in boxes),
                min(int(box[2]) for box in boxes),
                max(int(box[3]) for box in boxes),
            ]
            if not _edge_microtext_box(envelope, video):
                continue

            text_occurrences: dict[str, list[int]] = {}
            for event in component:
                text = _normalized_review_text(event.get("ocr_best_text"))
                if len(text) < 4:
                    continue
                text_occurrences.setdefault(text, []).append(
                    int(event.get("start_ms") or 0)
                )
            repeated_across_time = any(
                len(times) >= 3 and max(times) - min(times) >= 5_000
                for times in text_occurrences.values()
            )
            meaningful_counts = sorted(
                (len(times) for times in text_occurrences.values()),
                reverse=True,
            )
            meaningful_total = sum(meaningful_counts)
            dominant_ratio = (
                sum(meaningful_counts[:2]) / meaningful_total
                if meaningful_total
                else 0.0
            )
            if (
                not repeated_across_time
                or meaningful_total < 5
                or dominant_ratio < 0.60
            ):
                continue

            cluster_index = len(accepted)
            accepted.append(component)
            for event in component:
                track_id = str(event.get("track_id") or "")
                if track_id:
                    cluster_by_track[track_id] = cluster_index
    return accepted, cluster_by_track


def _semantic_review_profile(
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Consolidate exact mask tracks into conservative Gemini review groups.

    This never changes the product/exact event graph.  It only reduces Gemini's
    reasoning granularity; decisions are expanded back to exact track IDs and
    validated against the original profile before STTN selection.
    """
    events = sorted(
        [dict(event) for event in profile.get("events") or []],
        key=lambda event: (
            int(event.get("start_ms") or 0),
            int(event.get("end_ms") or 0),
            str(event.get("track_id") or ""),
        ),
    )
    if not events:
        semantic = dict(profile)
        semantic["events"] = []
        semantic["event_count"] = 0
        return semantic, {}

    video = dict(profile.get("video") or {})
    persistent_clusters: list[list[dict[str, Any]]] = []
    for candidate in events:
        if not _persistent_edge_candidate(candidate, video):
            continue
        target = next(
            (
                cluster for cluster in persistent_clusters
                if _review_texts_match(
                    _semantic_identity_text(cluster[0]), _semantic_identity_text(candidate)
                )
                and max(
                    _review_box_affinity(member.get("bbox"), candidate.get("bbox"))
                    for member in cluster
                ) >= 0.72
            ),
            None,
        )
        (target if target is not None else persistent_clusters.append([]) or persistent_clusters[-1]).append(candidate)
    persistent_cluster_by_track: dict[str, int] = {}
    for cluster_index, cluster in enumerate(persistent_clusters):
        ranges = sorted(
            (int(item.get("start_ms") or 0), int(item.get("end_ms") or item.get("start_ms") or 0))
            for item in cluster
        )
        merged_ranges: list[list[int]] = []
        for start_ms, end_ms in ranges:
            if merged_ranges and start_ms <= merged_ranges[-1][1]:
                merged_ranges[-1][1] = max(merged_ranges[-1][1], end_ms)
            else:
                merged_ranges.append([start_ms, end_ms])
        span_ms = max(1, ranges[-1][1] - ranges[0][0]) if ranges else 1
        coverage = sum(end - start for start, end in merged_ranges) / span_ms
        if len(cluster) >= 3 and span_ms >= 5_000 and coverage >= 0.30:
            for item in cluster:
                persistent_cluster_by_track[str(item.get("track_id") or "")] = cluster_index
    visual_superclusters, visual_supercluster_by_track = (
        _persistent_edge_visual_superclusters(events, video)
    )
    groups: list[list[dict[str, Any]]] = [list(cluster) for cluster in visual_superclusters]
    visual_assigned_ids = {
        str(event.get("track_id") or "")
        for cluster in visual_superclusters
        for event in cluster
    }
    for event in events:
        if str(event.get("track_id") or "") in visual_assigned_ids:
            continue
        best_group = None
        best_score = 0.0
        event_start = int(event.get("start_ms") or 0)
        event_end = int(event.get("end_ms") or event_start)
        event_text = _semantic_identity_text(event)
        # Persistent edge overlays can be separated by many unrelated caption
        # groups, so their semantic owner must remain searchable globally.
        for group in reversed(groups):
            anchor = group[-1]
            anchor_start = int(anchor.get("start_ms") or 0)
            anchor_end = int(anchor.get("end_ms") or anchor_start)
            gap_ms = max(0, event_start - anchor_end, anchor_start - event_end)
            affinity = _review_box_affinity(anchor.get("bbox"), event.get("bbox"))
            anchor_text = _semantic_identity_text(anchor)
            normalized_event_text = _normalized_review_text(event_text)
            text_match = _review_texts_match(anchor_text, normalized_event_text)
            stable_repeated_match = _stable_repeated_text_owner_match(
                anchor,
                event,
                affinity,
            )
            persistent_match = bool(
                _review_texts_match(anchor_text, normalized_event_text)
                and persistent_cluster_by_track.get(str(anchor.get("track_id") or ""))
                == persistent_cluster_by_track.get(str(event.get("track_id") or ""))
                and str(anchor.get("track_id") or "") in persistent_cluster_by_track
            )
            if gap_ms > 420 and not persistent_match and not stable_repeated_match:
                continue
            if affinity < 0.34 and not persistent_match and not stable_repeated_match:
                continue
            short_fragment = bool(
                min(
                    float(anchor.get("duration_seconds") or 0.0),
                    float(event.get("duration_seconds") or 0.0),
                )
                < 0.5
                and gap_ms <= 120
                and affinity >= 0.56
                and bool(anchor_text) != bool(normalized_event_text)
                and _review_shared_frames(anchor, event) > 0
            )
            if not text_match and not short_fragment:
                continue
            gap_penalty = (
                0.0
                if stable_repeated_match
                else gap_ms / (30_000.0 if persistent_match else 2_000.0)
            )
            score = (
                affinity
                + (0.65 if text_match else 0.0)
                + (0.20 if stable_repeated_match else 0.0)
                - gap_penalty
            )
            if score > best_score:
                best_group = group
                best_score = score
        if best_group is None:
            groups.append([event])
        else:
            best_group.append(event)

    semantic_events: list[dict[str, Any]] = []
    children_by_semantic_id: dict[str, list[str]] = {}
    for index, group in enumerate(groups, start=1):
        children = [str(event.get("track_id") or "") for event in group]
        semantic_id = children[0] if len(group) == 1 else f"sg_{index:06d}"
        children_by_semantic_id[semantic_id] = children
        representative = max(
            group,
            key=lambda event: (
                float(event.get("ocr_best_confidence") or 0.0),
                int(event.get("observed_frames") or 0),
                float(event.get("duration_seconds") or 0.0),
            ),
        )
        start_ms = min(int(event.get("start_ms") or 0) for event in group)
        end_ms = max(int(event.get("end_ms") or 0) for event in group)
        start_frame = min(int(event.get("start_frame") or 0) for event in group)
        end_frame = max(int(event.get("end_frame") or 0) for event in group)
        effective_start_frame = min(
            int(event.get("effective_start_frame") or event.get("start_frame") or 0)
            for event in group
        )
        effective_end_frame = max(
            int(event.get("effective_end_frame") or event.get("end_frame") or 0)
            for event in group
        )
        effective_start_ms = min(
            int(
                event.get("effective_start_ms")
                if event.get("effective_start_ms") is not None
                else event.get("start_ms") or 0
            )
            for event in group
        )
        effective_end_ms = max(
            int(
                event.get("effective_end_ms")
                if event.get("effective_end_ms") is not None
                else event.get("end_ms") or 0
            )
            for event in group
        )
        combined_frame_boxes = [
            dict(observation)
            for event in group
            for observation in event.get("frame_boxes") or []
            if isinstance(observation, dict)
        ]
        combined_sources = sorted(
            {
                str(source)
                for event in group
                for source in event.get("detection_sources") or []
                if str(source)
            }
        )
        candidate_by_text: dict[str, dict[str, Any]] = {}
        for event in group:
            for candidate in event.get("ocr_candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                text = str(candidate.get("text") or "")
                previous = candidate_by_text.get(text)
                if previous is None or float(candidate.get("confidence") or 0.0) > float(
                    previous.get("confidence") or 0.0
                ):
                    candidate_by_text[text] = dict(candidate)
        semantic_event = {
            **representative,
            "track_id": semantic_id,
            "ocr_best_text": _semantic_identity_text(representative)
            or representative.get("ocr_best_text")
            or "",
            "semantic_child_track_ids": children,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_seconds": round(max(0, end_ms - start_ms) / 1000.0, 6),
            "effective_start_frame": effective_start_frame,
            "effective_end_frame": effective_end_frame,
            "effective_start_ms": effective_start_ms,
            "effective_end_ms": effective_end_ms,
            "effective_duration_seconds": round(
                max(0, effective_end_ms - effective_start_ms) / 1000.0,
                6,
            ),
            "eligible_for_auto_translation": True,
            "observed_frames": len(
                {int(item.get("frame_no") or 0) for item in combined_frame_boxes}
            ),
            "frame_boxes": sorted(
                combined_frame_boxes,
                key=lambda item: (
                    int(item.get("frame_no") or 0),
                    (item.get("bbox") or [0, 0, 0, 0])[2],
                    (item.get("bbox") or [0, 0, 0, 0])[0],
                ),
            ),
            "detection_sources": combined_sources,
            "ocr_candidates": sorted(
                candidate_by_text.values(),
                key=lambda item: -float(item.get("confidence") or 0.0),
            ),
        }
        visual_cluster_ids = {
            visual_supercluster_by_track.get(str(event.get("track_id") or ""))
            for event in group
            if str(event.get("track_id") or "") in visual_supercluster_by_track
        }
        visual_persistent_supertrack = bool(
            len(visual_cluster_ids) == 1
            and len(group) >= 8
            and all(
                str(event.get("track_id") or "") in visual_supercluster_by_track
                for event in group
            )
        )
        persistent_fixed_overlay = bool(
            visual_persistent_supertrack
            or (
                len(group) >= 3
                and end_ms - start_ms >= 5_000
                and all(_persistent_edge_candidate(event, video) for event in group)
                and all(
                    _review_texts_match(group[0].get("ocr_best_text"), event.get("ocr_best_text"))
                    for event in group[1:]
                )
            )
        )
        if visual_persistent_supertrack:
            text_counts: dict[str, tuple[int, str]] = {}
            for event in group:
                original_text = str(event.get("ocr_best_text") or "").strip()
                normalized_text = _normalized_review_text(original_text)
                if len(normalized_text) < 4:
                    continue
                count, best_original = text_counts.get(normalized_text, (0, original_text))
                if len(original_text) > len(best_original):
                    best_original = original_text
                text_counts[normalized_text] = (count + 1, best_original)
            dominant_texts = [
                original
                for _normalized, (count, original) in sorted(
                    text_counts.items(),
                    key=lambda item: (-item[1][0], -len(item[0]), item[0]),
                )
                if count >= 3
            ][:2]
            if dominant_texts:
                semantic_event["ocr_best_text"] = " / ".join(dominant_texts)
            semantic_event["persistent_visual_supertrack"] = True
            semantic_event["persistent_visual_child_count"] = len(group)
        if persistent_fixed_overlay:
            boxes = [event.get("bbox") or [] for event in group]
            semantic_event["persistent_fixed_overlay"] = True
            semantic_event["persistent_bbox"] = [
                min(int(box[0]) for box in boxes), max(int(box[1]) for box in boxes),
                min(int(box[2]) for box in boxes), max(int(box[3]) for box in boxes),
            ]
        semantic_events.append(semantic_event)

    semantic = dict(profile)
    semantic["events"] = semantic_events
    semantic["event_count"] = len(semantic_events)
    semantic["semantic_consolidation"] = {
        "exact_event_count": len(events),
        "semantic_event_count": len(semantic_events),
        "persistent_visual_supertrack_count": len(visual_superclusters),
        "persistent_visual_child_count": sum(len(cluster) for cluster in visual_superclusters),
    }
    return semantic, children_by_semantic_id


def _semantic_fragmentation_guard_metrics(
    profile: dict[str, Any],
    *,
    exact_count: int,
    semantic_count: int,
) -> dict[str, Any]:
    """Detect event-graph explosions before they spend many strong Gemini calls."""
    batch_count = (
        math.ceil(semantic_count / GEMINI_REVIEW_BATCH_EVENTS)
        if semantic_count
        else 0
    )
    try:
        duration_seconds = float(
            (profile.get("video") or {}).get("duration_seconds") or 0.0
        )
    except (TypeError, ValueError):
        duration_seconds = 0.0
    if duration_seconds <= 0.0:
        duration_seconds = max(
            [
                float(event.get("end_ms") or 0.0) / 1000.0
                for event in profile.get("events") or []
            ]
            or [0.0]
        )
    retention_ratio = semantic_count / max(1, exact_count)
    events_per_second = semantic_count / max(1.0, duration_seconds)
    blocked = bool(
        batch_count >= SEMANTIC_FRAGMENTATION_GUARD_MIN_BATCHES
        and events_per_second >= SEMANTIC_FRAGMENTATION_GUARD_MAX_EVENTS_PER_SECOND
        and retention_ratio >= SEMANTIC_FRAGMENTATION_GUARD_MIN_RETENTION_RATIO
    )
    return {
        "blocked": blocked,
        "batch_count": batch_count,
        "duration_seconds": round(duration_seconds, 3),
        "retention_ratio": round(retention_ratio, 4),
        "events_per_second": round(events_per_second, 4),
    }


def _expand_semantic_decisions(
    *,
    original_profile: dict[str, Any],
    semantic_decisions: list[dict[str, Any]],
    children_by_semantic_id: dict[str, list[str]],
    config: object,
) -> list[dict[str, Any]]:
    response_items: list[dict[str, Any]] = []
    for decision in semantic_decisions:
        semantic_id = str(decision.get("track_id") or "")
        children = children_by_semantic_id.get(semantic_id) or []
        for child_track_id in children:
            response_items.append(
                {
                    "track_id": child_track_id,
                    "classification": decision.get("classification"),
                    "text_origin": decision.get("text_origin"),
                    "source_text_corrected": str(
                        decision.get("source_text_corrected") or ""
                    ),
                    "vietnamese_translation": str(
                        decision.get("vietnamese_translation") or ""
                    ),
                    "highlight_terms": list(decision.get("highlight_terms") or []),
                    "replacement_action": decision.get("replacement_action"),
                    "confidence": decision.get("confidence"),
                    "reason_vi": str(decision.get("reason_vi") or ""),
                }
            )
    exact_decisions = validate_gemini_decisions(
        original_profile,
        {"items": response_items},
        config,
    )["decisions"]
    exact_by_id = {
        str(item.get("track_id") or ""): item
        for item in exact_decisions
    }
    event_by_id = {
        str(event.get("track_id") or ""): event
        for event in original_profile.get("events") or []
    }
    for semantic_decision in semantic_decisions:
        semantic_id = str(semantic_decision.get("track_id") or "")
        children = children_by_semantic_id.get(semantic_id) or []
        for child_track_id in children:
            child = exact_by_id.get(child_track_id)
            if child is not None:
                if len(children) > 1:
                    child["semantic_review_track_id"] = semantic_id
                    child["semantic_child_count"] = len(children)
                    if semantic_decision.get("persistent_fixed_overlay"):
                        child["persistent_fixed_overlay"] = True
                        child["persistent_bbox"] = list(semantic_decision.get("persistent_bbox") or child.get("bbox") or [])
                    if semantic_decision.get("persistent_visual_supertrack"):
                        child["persistent_visual_supertrack"] = True
                        child["persistent_visual_child_count"] = int(
                            semantic_decision.get("persistent_visual_child_count")
                            or len(children)
                        )
                # Keep the evidence from the targeted persistent multiline
                # repair when the semantic decision is expanded back to exact
                # OCR tracks.  `validate_gemini_decisions` intentionally drops
                # unknown provider fields, but this marker is local routing
                # evidence used by the reconciliation guard below.
                if semantic_decision.get("multiline_repair_evidence"):
                    child["multiline_repair_evidence"] = True
                    child["routing_override"] = str(
                        semantic_decision.get("routing_override")
                        or "persistent_multiline_row_repair"
                    )
        if len(children) <= 1:
            continue
        if semantic_decision.get("render_action") != "replace_with_vietnamese":
            continue
        ordered_children = sorted(
            children,
            key=lambda track_id: (
                int((exact_by_id.get(track_id) or {}).get("start_ms") or 0),
                int((exact_by_id.get(track_id) or {}).get("end_ms") or 0),
                str(track_id),
            ),
        )
        if semantic_decision.get("persistent_fixed_overlay"):
            render_components = [ordered_children]
        else:
            render_components: list[list[str]] = []
            active_component_end_ms = -1
            for child_track_id in ordered_children:
                child = exact_by_id.get(child_track_id) or {}
                child_start_ms = int(child.get("start_ms") or 0)
                child_end_ms = int(child.get("end_ms") or child_start_ms)
                if (
                    render_components
                    and child_start_ms
                    <= active_component_end_ms + SEMANTIC_RENDER_COMPONENT_MAX_GAP_MS
                ):
                    render_components[-1].append(child_track_id)
                    active_component_end_ms = max(active_component_end_ms, child_end_ms)
                else:
                    render_components.append([child_track_id])
                    active_component_end_ms = child_end_ms

        component_count = len(render_components)
        for component_index, component_children in enumerate(render_components, start=1):
            render_anchor_id = max(
                component_children,
                key=lambda track_id: (
                    float((event_by_id.get(track_id) or {}).get("duration_seconds") or 0.0),
                    int((event_by_id.get(track_id) or {}).get("observed_frames") or 0),
                    float((event_by_id.get(track_id) or {}).get("ocr_best_confidence") or 0.0),
                ),
            )
            if semantic_decision.get("persistent_fixed_overlay"):
                component_start_ms = int(semantic_decision.get("start_ms") or 0)
                component_end_ms = int(
                    semantic_decision.get("end_ms") or component_start_ms
                )
                component_start_frame = int(semantic_decision.get("start_frame") or 0)
                component_end_frame = int(
                    semantic_decision.get("end_frame") or component_start_frame
                )
            else:
                component_start_ms = min(
                    int(exact_by_id[track_id].get("start_ms") or 0)
                    for track_id in component_children
                )
                component_end_ms = max(
                    int(
                        exact_by_id[track_id].get("end_ms")
                        or exact_by_id[track_id].get("start_ms")
                        or 0
                    )
                    for track_id in component_children
                )
                component_start_frame = min(
                    int(exact_by_id[track_id].get("start_frame") or 0)
                    for track_id in component_children
                )
                component_end_frame = max(
                    int(
                        exact_by_id[track_id].get("end_frame")
                        or exact_by_id[track_id].get("start_frame")
                        or 0
                    )
                    for track_id in component_children
                )

            for child_track_id in component_children:
                child = exact_by_id[child_track_id]
                child["selected_for_inpaint"] = True
                child["semantic_render_component_index"] = component_index
                child["semantic_render_component_count"] = component_count
                if child_track_id == render_anchor_id:
                    child["render_action"] = "replace_with_vietnamese"
                    child["auto_approved"] = True
                    child["eligible_for_auto_translation"] = True
                    child["start_ms"] = component_start_ms
                    child["end_ms"] = component_end_ms
                    child["start_frame"] = component_start_frame
                    child["end_frame"] = component_end_frame
                    child["duration_seconds"] = round(
                        max(0, component_end_ms - component_start_ms) / 1000.0,
                        6,
                    )
                    child["render_anchor_for_semantic"] = True
                else:
                    child["render_action"] = "remove_original"
                    child["auto_approved"] = False
                    child["routing_override"] = "semantic_child_fragment"
                    child["render_anchor_track_id"] = render_anchor_id
    return exact_decisions


def _extend_confirmed_persistent_supertrack_tail(
    profile: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> int:
    """Extend a confirmed persistent edge super-track through a short OCR tail miss."""
    video = dict(profile.get("video") or {})
    fps = max(1.0, float(video.get("fps") or 0.0))
    frame_count = int(video.get("frame_count") or 0)
    if frame_count <= 0:
        return 0

    maximum_tail_gap = max(1, int(round(fps * 1.5)))
    minimum_span = max(1, int(round(fps * 8.0)))
    video_end_ms = int(round(frame_count * 1000.0 / fps))
    extended = 0
    for anchor in decisions:
        if not (
            bool(anchor.get("selected_for_inpaint"))
            and bool(anchor.get("persistent_fixed_overlay"))
            and bool(anchor.get("persistent_visual_supertrack"))
            and bool(anchor.get("render_anchor_for_semantic"))
            and anchor.get("text_origin") == "post_production_overlay"
            and anchor.get("render_action") == "replace_with_vietnamese"
            and float(anchor.get("confidence") or 0.0) >= 0.90
            and int(anchor.get("semantic_child_count") or 0) >= 8
            and _edge_microtext_box(
                anchor.get("persistent_bbox") or anchor.get("bbox"), video
            )
        ):
            continue
        start_frame = max(1, int(anchor.get("start_frame") or 1))
        end_frame = max(start_frame, int(anchor.get("end_frame") or start_frame))
        tail_gap = frame_count - end_frame
        if tail_gap <= 0 or tail_gap > maximum_tail_gap:
            continue
        if end_frame - start_frame + 1 < minimum_span:
            continue

        anchor["persistent_tail_original_end_frame"] = end_frame
        anchor["persistent_tail_original_end_ms"] = int(anchor.get("end_ms") or 0)
        anchor["end_frame"] = frame_count
        anchor["effective_end_frame"] = frame_count
        anchor["end_ms"] = max(int(anchor.get("end_ms") or 0), video_end_ms)
        anchor["effective_end_ms"] = max(
            int(anchor.get("effective_end_ms") or 0), video_end_ms
        )
        anchor["duration_seconds"] = round(
            max(0, int(anchor["end_ms"]) - int(anchor.get("start_ms") or 0))
            / 1000.0,
            6,
        )
        anchor["persistent_tail_extended"] = True
        anchor["persistent_tail_extension_frames"] = tail_gap
        extended += 1
    return extended


def _suppress_nested_source_fragment_renders(
    decisions: list[dict[str, Any]],
) -> int:
    """Suppress a strict source-text fragment inside a larger approved overlay.

    Animated editor headlines can temporarily be OCR'd both as the full phrase
    and as a short substring. If both are independently approved, drawing both
    Vietnamese translations creates a duplicate label. This changes rendering
    ownership only; exact masks remain selected for cleanup.
    """
    candidates = [
        item
        for item in decisions
        if bool(item.get("selected_for_inpaint"))
        and item.get("render_action") == "replace_with_vietnamese"
        and item.get("text_origin") == "post_production_overlay"
        and float(item.get("confidence") or 0.0) >= 0.90
    ]
    suppressed = 0
    for child in candidates:
        child_text = _normalized_review_text(child.get("source_text_corrected"))
        child_box = [int(value) for value in child.get("bbox") or []]
        if not child_text or len(child_box) != 4:
            continue
        # A one-Han-glyph animation state is still a semantic fragment. The
        # previous len>=2 shortcut let frames such as 上 -> "Đi" become an
        # independent Vietnamese render while the complete headline existed.
        # Arbitrary one-character UI labels are protected by the stricter
        # semantic/geometry/timeline ownership checks below.
        single_han_fragment = bool(
            len(child_text) == 1
            and any(
                "\u3400" <= char <= "\u4dbf"
                or "\u4e00" <= char <= "\u9fff"
                or "\uf900" <= char <= "\ufaff"
                for char in child_text
            )
        )
        if len(child_text) < 2 and not single_han_fragment:
            continue
        child_start = int(child.get("start_frame") or 0)
        child_end = max(child_start, int(child.get("end_frame") or child_start))
        child_span = max(1, child_end - child_start + 1)
        child_area = max(1, (child_box[1] - child_box[0]) * (child_box[3] - child_box[2]))
        best_parent = None
        best_score = (-1, -1.0, -1)
        for parent in candidates:
            if parent is child:
                continue
            parent_text = _normalized_review_text(parent.get("source_text_corrected"))
            if len(parent_text) < len(child_text) + 2 or child_text not in parent_text:
                continue
            if str(parent.get("classification") or "") != str(child.get("classification") or ""):
                continue
            if str(parent.get("design_role") or "neutral") != str(child.get("design_role") or "neutral"):
                continue
            parent_box = [int(value) for value in parent.get("bbox") or []]
            if len(parent_box) != 4:
                continue
            parent_area = max(1, (parent_box[1] - parent_box[0]) * (parent_box[3] - parent_box[2]))
            minimum_area_ratio = 4.0 if single_han_fragment else 2.0
            if parent_area < child_area * minimum_area_ratio:
                continue
            if _box_covers_target(parent_box, child_box) < 0.55:
                continue
            parent_start = int(parent.get("start_frame") or 0)
            parent_end = max(parent_start, int(parent.get("end_frame") or parent_start))
            parent_span = max(1, parent_end - parent_start + 1)
            overlap = max(0, min(child_end, parent_end) - max(child_start, parent_start) + 1)
            minimum_timeline_overlap = 0.90 if single_han_fragment else 0.60
            if overlap / child_span < minimum_timeline_overlap:
                continue
            if single_han_fragment and not (
                parent_start <= child_start
                and parent_end >= child_end
                and parent_span >= child_span * 2
            ):
                continue
            score = (len(parent_text), float(parent.get("confidence") or 0.0), parent_area)
            if score > best_score:
                best_parent = parent
                best_score = score
        if best_parent is None:
            continue
        child["render_action"] = "remove_original"
        child["auto_approved"] = False
        child["routing_override"] = "nested_source_fragment_render_suppressed"
        child["render_anchor_track_id"] = str(best_parent.get("track_id") or "")
        child["reason_vi"] = (
            str(child.get("reason_vi") or "").strip()
            + " Chan render fragment long nhau: source text la chuoi con cua "
            f"overlay hau ky lon hon {best_parent.get('track_id')}; van giu exact "
            "mask de xoa chu goc nhung khong render ban dich thu hai."
        ).strip()
        suppressed += 1
    return suppressed


def _canonicalize_repeated_overlay_translations(
    decisions: list[dict[str, Any]],
) -> int:
    """Use one Vietnamese translation for repeated reviewed editor text.

    Gemini can translate the same short label differently in separate review
    batches. Only confident post-production replacements with the same
    corrected source text, classification, and design role share a canonical
    translation. Physical/scene text, spoken subtitles, uncertain items, and
    different semantic roles are intentionally excluded.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for decision in decisions:
        if not (
            bool(decision.get("selected_for_inpaint"))
            and decision.get("render_action") == "replace_with_vietnamese"
            and decision.get("text_origin") == "post_production_overlay"
            and str(decision.get("classification") or "") in SILENT_HARD_SUB_CLASSES
            and float(decision.get("confidence") or 0.0) >= 0.90
        ):
            continue
        source_key = _normalized_review_text(decision.get("source_text_corrected"))
        translation = str(decision.get("vietnamese_translation") or "").strip()
        if not source_key or not translation:
            continue
        key = (
            source_key,
            str(decision.get("classification") or ""),
            str(decision.get("design_role") or "neutral"),
        )
        groups.setdefault(key, []).append(decision)

    overridden = 0
    for (source_key, _classification, _design_role), group in groups.items():
        if len(group) < 2:
            continue
        translation_buckets: dict[str, list[dict[str, Any]]] = {}
        for item in group:
            normalized_translation = re.sub(
                r"\s+",
                " ",
                str(item.get("vietnamese_translation") or "").strip(),
            ).casefold()
            if normalized_translation:
                translation_buckets.setdefault(normalized_translation, []).append(item)
        if len(translation_buckets) <= 1:
            continue
        canonical_bucket = max(
            translation_buckets.values(),
            key=lambda items: (
                len(items),
                sum(float(item.get("confidence") or 0.0) for item in items),
                max(float(item.get("confidence") or 0.0) for item in items),
                sum(float(item.get("duration_seconds") or 0.0) for item in items),
            ),
        )
        canonical = max(
            canonical_bucket,
            key=lambda item: (
                float(item.get("confidence") or 0.0),
                float(item.get("duration_seconds") or 0.0),
                -int(item.get("start_ms") or 0),
                str(item.get("track_id") or ""),
            ),
        )
        canonical_translation = str(
            canonical.get("vietnamese_translation") or ""
        ).strip()
        canonical_highlights = _validated_highlight_terms(
            canonical_translation,
            canonical.get("highlight_terms") or [],
        )
        canonical_track_id = str(canonical.get("track_id") or "")
        for item in group:
            previous = str(item.get("vietnamese_translation") or "").strip()
            if previous == canonical_translation:
                continue
            item["translation_consistency_original"] = previous
            item["vietnamese_translation"] = canonical_translation
            item["highlight_terms"] = list(canonical_highlights)
            item["translation_consistency_source_track_id"] = canonical_track_id
            item["translation_consistency_source_key"] = source_key
            overridden += 1
    return overridden


def _deduplicate_vietnamese_render_fragments(
    decisions: list[dict[str, Any]],
) -> int:
    """Keep one Vietnamese render anchor for duplicate semantic fragments.

    OCR can occasionally split one persistent overlay into overlapping text
    events even after recognition-based merging.  Every child still needs its
    exact masks selected for inpaint, but drawing the same translation for
    every child creates duplicate Vietnamese text.  This final semantic guard
    only changes rendering ownership; it never deselects an inpaint mask.
    """
    candidates = sorted(
        [
            decision
            for decision in decisions
            if decision.get("render_action") == "replace_with_vietnamese"
            and bool(decision.get("selected_for_inpaint"))
        ],
        key=lambda item: (
            int(item.get("start_ms") or 0),
            int(item.get("end_ms") or 0),
            str(item.get("track_id") or ""),
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    for decision in candidates:
        start_ms = int(decision.get("start_ms") or 0)
        end_ms = int(decision.get("end_ms") or start_ms)
        best_group = None
        best_affinity = 0.0
        for group in reversed(groups[-32:]):
            for member in group:
                member_start = int(member.get("start_ms") or 0)
                member_end = int(member.get("end_ms") or member_start)
                gap_ms = max(
                    0,
                    start_ms - member_end,
                    member_start - end_ms,
                )
                if gap_ms > 420:
                    continue
                if not _review_texts_match(
                    member.get("source_text_corrected"),
                    decision.get("source_text_corrected"),
                ):
                    continue
                affinity = _review_box_affinity(
                    member.get("bbox"),
                    decision.get("bbox"),
                )
                if affinity < 0.72:
                    continue
                if affinity > best_affinity:
                    best_group = group
                    best_affinity = affinity
        if best_group is None:
            groups.append([decision])
        else:
            best_group.append(decision)

    suppressed = 0
    for group in groups:
        if len(group) <= 1:
            continue
        anchor = max(
            group,
            key=lambda item: (
                int(item.get("end_ms") or 0) - int(item.get("start_ms") or 0),
                float(item.get("confidence") or 0.0),
                str(item.get("track_id") or ""),
            ),
        )
        child_ids = [str(item.get("track_id") or "") for item in group]
        anchor_id = str(anchor.get("track_id") or "")
        start_ms = min(int(item.get("start_ms") or 0) for item in group)
        end_ms = max(int(item.get("end_ms") or start_ms) for item in group)
        start_frame = min(int(item.get("start_frame") or 0) for item in group)
        end_frame = max(
            int(item.get("end_frame") or item.get("start_frame") or 0)
            for item in group
        )
        anchor["start_ms"] = start_ms
        anchor["end_ms"] = end_ms
        anchor["start_frame"] = start_frame
        anchor["end_frame"] = end_frame
        anchor["duration_seconds"] = round(max(0, end_ms - start_ms) / 1000.0, 6)
        anchor["render_anchor_for_duplicate_text"] = True
        anchor["duplicate_render_child_track_ids"] = child_ids
        for child in group:
            if child is anchor:
                continue
            child["render_action"] = "remove_original"
            child["auto_approved"] = False
            child["routing_override"] = "duplicate_vietnamese_render_fragment"
            child["render_anchor_track_id"] = anchor_id
            suppressed += 1
    return suppressed


def _reconcile_persistent_multiline_fragments(
    profile: dict[str, Any], decisions: list[dict[str, Any]]
) -> int:
    """Keep two reviewed rows when OCR repeatedly merged their source boxes.

    A short first-row track at each edge can be bridged by a long exact track
    whose OCR still reads the first row, even though Strong Review identifies
    its translation as the second row.  The source boxes remain intact for
    mask ownership; only render ownership, timing and placement change.
    """
    event_by_id = {
        str(event.get("track_id") or ""): event
        for event in profile.get("events") or []
    }

    def source_times(item: dict[str, Any]) -> tuple[int, int]:
        # Earlier envelope routing may stretch render times across the clip.
        # Use the OCR track's observed interval to identify edge fragments.
        event = event_by_id.get(str(item.get("track_id") or "")) or {}
        start = event.get("start_ms")
        end = event.get("end_ms")
        return (
            int(start if start is not None else item.get("start_ms") or 0),
            int(end if end is not None else item.get("end_ms") or 0),
        )
    active = [
        item for item in decisions
        if item.get("render_action") == "replace_with_vietnamese"
        and bool(item.get("selected_for_inpaint"))
        and item.get("text_origin") == "post_production_overlay"
    ]
    reconciled = 0
    used_ids: set[str] = set()
    for first in active:
        first_id = str(first.get("track_id") or "")
        if first_id in used_ids:
            continue
        first_start, first_end = source_times(first)
        for tail in active:
            tail_id = str(tail.get("track_id") or "")
            if tail_id in used_ids or tail is first:
                continue
            tail_start, tail_end = source_times(tail)
            if tail_start <= first_end or tail_end <= tail_start:
                continue
            if not _review_texts_match(
                first.get("source_text_corrected"), tail.get("source_text_corrected")
            ) or not _review_texts_match(
                first.get("vietnamese_translation"), tail.get("vietnamese_translation")
            ):
                continue
            if _review_box_affinity(first.get("bbox"), tail.get("bbox")) < 0.95:
                continue
            for bridge in active:
                bridge_id = str(bridge.get("track_id") or "")
                if bridge_id in used_ids or bridge is first or bridge is tail:
                    continue
                bridge_start, bridge_end = source_times(bridge)
                if (bridge_start > first_end + 420
                    or bridge_end < tail_start - 420
                    or bridge_end - bridge_start < (tail_end - first_start) * 0.6):
                    continue
                if _review_box_affinity(first.get("bbox"), bridge.get("bbox")) < 0.95:
                    continue
                if _review_texts_match(
                    first.get("source_text_corrected"), bridge.get("source_text_corrected")
                ):
                    continue
                # The reviewed second-row description and independent OCR
                # evidence are both required. A changing single-line label
                # must continue to fail the normal collision guard.
                reason = str(bridge.get("reason_vi") or "").casefold()
                if not bridge.get("multiline_repair_evidence") and not any(
                    token in reason for token in ("dòng thứ hai", "dòng 2")
                ):
                    continue
                ocr_text = _normalized_review_text(
                    (event_by_id.get(bridge_id) or {}).get("ocr_best_text")
                )
                first_text = _normalized_review_text(first.get("source_text_corrected"))
                bridge_text = _normalized_review_text(bridge.get("source_text_corrected"))
                if (
                    not bridge.get("multiline_repair_evidence")
                    and (
                        not ocr_text
                        or not first_text
                        or SequenceMatcher(None, ocr_text, first_text).ratio() < 0.65
                        or SequenceMatcher(None, ocr_text, bridge_text).ratio() > 0.35
                    )
                ):
                    continue
                box = [int(value) for value in first.get("bbox") or []]
                if len(box) != 4 or box[3] - box[2] < 40:
                    continue
                midpoint = (box[2] + box[3]) // 2
                first["render_target_bbox"] = [box[0], box[1], box[2], midpoint]
                bridge["render_target_bbox"] = [box[0], box[1], midpoint, box[3]]
                for item in (first, bridge):
                    item["multiline_render_group"] = first_id
                    item["start_ms"] = first_start
                    item["end_ms"] = tail_end
                    item["start_frame"] = min(
                        int(first.get("start_frame") or 0),
                        int(bridge.get("start_frame") or 0),
                    )
                    item["end_frame"] = max(
                        int(tail.get("end_frame") or 0),
                        int(bridge.get("end_frame") or 0),
                    )
                    item["semantic_start_frame"] = item["start_frame"]
                    item["semantic_end_frame"] = item["end_frame"]
                    item["duration_seconds"] = round((tail_end - first_start) / 1000, 6)
                first["multiline_render_row"] = 0
                bridge["multiline_render_row"] = 1
                tail["render_action"] = "remove_original"
                tail["auto_approved"] = False
                tail["routing_override"] = "persistent_multiline_duplicate_fragment"
                tail["render_anchor_track_id"] = first_id
                third_rows = []
                for candidate in active:
                    if candidate is first or candidate is bridge or candidate is tail:
                        continue
                    candidate_box = [int(value) for value in candidate.get("bbox") or []]
                    if (
                        len(candidate_box) == 4
                        and abs(candidate_box[0] - box[0]) <= 18
                        and abs(candidate_box[1] - box[1]) <= 18
                        and -4 <= candidate_box[2] - box[3] <= 12
                        and abs(int(candidate.get("start_ms") or 0) - first_start) <= 80
                        and abs(int(candidate.get("end_ms") or 0) - tail_end) <= 80
                        and not _review_texts_match(
                            candidate.get("source_text_corrected"),
                            first.get("source_text_corrected"),
                        )
                        and not _review_texts_match(
                            candidate.get("source_text_corrected"),
                            bridge.get("source_text_corrected"),
                        )
                    ):
                        third_rows.append(candidate)
                if len(third_rows) == 1:
                    third_rows[0]["multiline_render_group"] = first_id
                    third_rows[0]["multiline_render_row"] = 2
                used_ids.update((first_id, bridge_id, tail_id))
                reconciled += 1
                break
            if first_id in used_ids:
                break
    return reconciled


def _persistent_multiline_bridge_groups(
    profile: dict[str, Any], decisions: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Find a narrow OCR pattern that needs a row-specific semantic reread."""
    active = [
        item for item in decisions
        if item.get("render_action") == "replace_with_vietnamese"
        and bool(item.get("selected_for_inpaint"))
        and item.get("text_origin") == "post_production_overlay"
    ]
    groups: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for first in active:
        first_text = first.get("source_text_corrected")
        for tail in active:
            if tail is first:
                continue
            if int(tail.get("start_ms") or 0) <= int(first.get("end_ms") or 0):
                continue
            if not _review_texts_match(first_text, tail.get("source_text_corrected")):
                continue
            if not _review_texts_match(
                first.get("vietnamese_translation"), tail.get("vietnamese_translation")
            ):
                continue
            if _review_box_affinity(first.get("bbox"), tail.get("bbox")) < 0.95:
                continue
            for bridge in active:
                if bridge is first or bridge is tail:
                    continue
                bridge_duration = int(bridge.get("end_ms") or 0) - int(bridge.get("start_ms") or 0)
                if bridge_duration < 5_000:
                    continue
                if _review_box_affinity(first.get("bbox"), bridge.get("bbox")) < 0.95:
                    continue
                if not _review_texts_match(first_text, bridge.get("source_text_corrected")):
                    continue
                if (int(bridge.get("start_ms") or 0) > int(first.get("end_ms") or 0) + 700
                    or int(bridge.get("end_ms") or 0) < int(tail.get("start_ms") or 0) - 700):
                    continue
                groups.append((first, bridge, tail))
                break
            if groups and groups[-1][0] is first:
                break
    return groups


def _repair_persistent_multiline_bridges(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    decisions: list[dict[str, Any]],
    clients: list[GeminiRestClient],
    config: dict[str, Any],
    work_folder: str | Path,
    srt_text: str,
    preferred_client_index: int,
    log: Callable[[str], None],
) -> list[str]:
    """Reread only proven multiline bridges, keeping model order 3.6 -> 3.5."""
    repaired_models: list[str] = []
    for first, bridge, tail in _persistent_multiline_bridge_groups(profile, decisions):
        bridge_id = str(bridge.get("track_id") or "")
        focused_profile = dict(profile)
        focused_profile["events"] = [
            event for event in profile.get("events") or []
            if str(event.get("track_id") or "") in {
                str(first.get("track_id") or ""), bridge_id,
                str(tail.get("track_id") or ""),
            }
        ]
        focused_profile["event_count"] = len(focused_profile["events"])
        sheets = build_contact_sheets(
            focused_profile,
            profile_path,
            Path(work_folder) / "contact-sheets" / "persistent-multiline-repair",
        )
        prompt = build_gemini_prompt(focused_profile, srt_text=srt_text) + (
            "\n\nĐÂY LÀ LƯỢT REPAIR CHO MỘT CỤM BA TRACK ĐÃ ĐƯỢC XÁC NHẬN. "
            f"Track đầu và cuối ({first.get('track_id')}, {tail.get('track_id')}) "
            f"là cùng một dòng trên. Track giữa {bridge_id} là dòng riêng ở ngay "
            "bên dưới, dù OCR của nó có thể lặp dòng trên. Chỉ đọc và dịch dòng "
            f"bên dưới cho {bridge_id}; tuyệt đối không trả lại source/translation "
            "của track đầu. Nếu ảnh không đủ rõ, trả uncertain cho track giữa."
        )
        resolved = False
        for model_name in GEMINI_REVIEW_MODELS:
            client_order = [
                *range(preferred_client_index, len(clients)),
                *range(0, preferred_client_index),
            ]
            for client_index in client_order:
                try:
                    response = clients[client_index].generate(
                        model=model_name,
                        prompt=prompt,
                        media=[],
                        inline_media_paths=sheets,
                        request_timeout_seconds=INLINE_REVIEW_TIMEOUT_SECONDS,
                        request_attempts=MODEL_503_RETRY_LIMIT,
                    )
                    candidate = validate_gemini_decisions(
                        focused_profile, response, config
                    )["decisions"]
                    candidate_bridge = next(
                        (item for item in candidate if str(item.get("track_id") or "") == bridge_id),
                        None,
                    )
                    if (
                        candidate_bridge
                        and candidate_bridge.get("replacement_action") == "replace_with_vietnamese"
                        and not _review_texts_match(
                            candidate_bridge.get("source_text_corrected"),
                            first.get("source_text_corrected"),
                        )
                    ):
                        bridge.update(candidate_bridge)
                        bridge["multiline_repair_evidence"] = True
                        bridge["routing_override"] = "persistent_multiline_row_repair"
                        repaired_models.append(model_name)
                        resolved = True
                        log(
                            f"Persistent multiline repair resolved {bridge_id} "
                            f"with {model_name}."
                        )
                        break
                except (GeminiRequestError, SilentTextLocalizationError) as exc:
                    log(
                        f"Persistent multiline repair {bridge_id} with {model_name} "
                        f"failed closed: {type(exc).__name__}."
                    )
            if resolved:
                break
    return repaired_models


def _conflicting_vietnamese_render_pairs(
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return temporally and spatially conflicting replacement anchors.

    Two different Vietnamese labels should never be drawn into essentially the
    same source box at the same time. This catches semantic groups whose render
    timeline was accidentally stretched across an intervening label. Same-text
    duplicates are handled separately by the dedupe guard.
    """
    candidates = [
        item
        for item in decisions
        if item.get("render_action") == "replace_with_vietnamese"
        and bool(item.get("selected_for_inpaint"))
    ]
    conflicts: list[dict[str, Any]] = []
    for first_index, first in enumerate(candidates):
        first_start = int(first.get("start_ms") or 0)
        first_end = int(first.get("end_ms") or first_start)
        for second in candidates[first_index + 1 :]:
            if (first.get("multiline_render_group")
                and first.get("multiline_render_group") == second.get("multiline_render_group")
                and first.get("multiline_render_row") != second.get("multiline_render_row")):
                continue
            if _review_texts_match(
                first.get("source_text_corrected"),
                second.get("source_text_corrected"),
            ):
                continue
            second_start = int(second.get("start_ms") or 0)
            second_end = int(second.get("end_ms") or second_start)
            overlap_ms = min(first_end, second_end) - max(first_start, second_start)
            if overlap_ms <= 0:
                continue
            affinity = _review_box_affinity(first.get("bbox"), second.get("bbox"))
            if affinity < 0.72:
                continue
            conflicts.append(
                {
                    "first_track_id": str(first.get("track_id") or ""),
                    "second_track_id": str(second.get("track_id") or ""),
                    "first_text": str(first.get("source_text_corrected") or ""),
                    "second_text": str(second.get("source_text_corrected") or ""),
                    "overlap_ms": int(overlap_ms),
                    "box_affinity": round(float(affinity), 6),
                }
            )
    return conflicts


def _parallel_lite_triage(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    source_video: Path,
    clients: list[GeminiRestClient],
    config: dict[str, Any],
    work_folder: str | Path,
    srt_text: str,
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run fail-closed Lite triage in parallel across independent projects.

    Lite may only finalize high-confidence keep decisions. Every possible
    removal and every uncertain result is forwarded to the strong model.
    """
    events = list(profile.get("events") or [])
    if not events:
        return [], {
            "enabled": True,
            "input_event_count": 0,
            "safe_keep_count": 0,
            "forwarded_to_strong_count": 0,
            "batch_count": 0,
            "worker_count": 0,
        }
    batch_size = int(config["lite_parallel_batch_size"])
    batches: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(events), batch_size), start=1):
        batch_profile = dict(profile)
        batch_profile["events"] = events[start : start + batch_size]
        batch_profile["event_count"] = len(batch_profile["events"])
        batches.append(batch_profile)
    worker_count = min(
        len(batches),
        len(clients),
        int(config["lite_parallel_workers"]),
    )
    if worker_count < 1:
        raise SilentTextLocalizationError("No Gemini client is available for Lite triage.")
    # Contact sheets are sufficient for this fail-closed first pass. Uploading
    # the full source video to every project would erase most of the latency
    # benefit. Missing temporal context can only cause forwarding to the strong
    # model; Lite is never allowed to remove or translate an event.
    active = list(enumerate(clients))
    primary = active[:worker_count]
    client_locks = [Lock() for _client in clients]

    def review_one(
        batch_number: int,
        batch_profile: dict[str, Any],
        client_info: tuple[int, GeminiRestClient],
    ) -> tuple[int, list[dict[str, Any]]]:
        client_index, gemini = client_info
        sheets = build_contact_sheets(
            batch_profile,
            profile_path,
            Path(work_folder) / "contact-sheets" / f"lite-{batch_number:03d}",
        )
        with client_locks[client_index]:
            triage_model = (
                ANTIGRAVITY_TRIAGE_MODEL
                if config["parallel_triage_provider"] == ANTIGRAVITY_TRIAGE_PROVIDER
                else str(config["parallel_triage_model"])
            )
            response = gemini.generate(
                model=triage_model,
                prompt=(
                    "This is a fast first-pass triage. Be conservative: "
                    "if the text could be an editorial overlay or evidence is "
                    "insufficient, return uncertain. Never guess scene text.\n\n"
                    + build_gemini_prompt(batch_profile, srt_text=srt_text)
                ),
                media=[],
                inline_media_paths=sheets,
                thinking_level="minimal",
                request_timeout_seconds=INLINE_REVIEW_TIMEOUT_SECONDS,
                request_attempts=1,
            )
            decisions = validate_gemini_decisions(
                batch_profile,
                response,
                config,
            )["decisions"]
            return batch_number, decisions

    results: dict[int, list[dict[str, Any]]] = {}
    failed_batches: list[tuple[int, dict[str, Any], int, Exception]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {}
        for batch_number, batch_profile in enumerate(batches, start=1):
            active_index = (batch_number - 1) % len(primary)
            future = executor.submit(
                review_one,
                batch_number,
                batch_profile,
                primary[active_index],
            )
            futures[future] = (batch_number, batch_profile, active_index)
        for future in as_completed(futures):
            batch_number, batch_profile, active_index = futures[future]
            try:
                results[batch_number] = future.result()[1]
            except Exception as exc:
                failed_batches.append(
                    (batch_number, batch_profile, active_index, exc)
                )
                log(
                    "Gemini Lite parallel batch failed on its assigned "
                    f"key/project: {batch_number}/{len(batches)} "
                    f"({type(exc).__name__}); trying a spare project."
                )
            else:
                log(
                    "Gemini Lite parallel triage batch completed: "
                    f"{batch_number}/{len(batches)}."
                )
    for batch_number, batch_profile, failed_index, _first_error in failed_batches:
        recovered = False
        failed_client_index = primary[failed_index][0]
        for offset in range(1, len(active) + 1):
            retry_index = (failed_client_index + offset) % len(active)
            if retry_index == failed_client_index:
                continue
            try:
                results[batch_number] = review_one(
                    batch_number,
                    batch_profile,
                    active[retry_index],
                )[1]
            except Exception as exc:
                log(
                    "Gemini Lite spare project failed for batch "
                    f"{batch_number}/{len(batches)} ({type(exc).__name__})."
                )
                continue
            recovered = True
            log(
                "Gemini Lite batch recovered on a spare project: "
                f"{batch_number}/{len(batches)}."
            )
            break
        if not recovered:
            log(
                "Gemini Lite batch unavailable; forwarding only this batch "
                f"to strong review: {batch_number}/{len(batches)}."
            )

    decisions = [
        item
        for batch_number in range(1, len(batches) + 1)
        for item in results.get(batch_number, [])
    ]
    keep_confidence = float(config["lite_safe_keep_confidence"])
    safe_keep: list[dict[str, Any]] = []
    for decision in decisions:
        if (
            decision.get("render_action") == "restore_original"
            # Lite is only allowed to close a clearly physical scene-text
            # case.  Short editorial labels (especially #action labels) were
            # being mistaken for decorative UI/watermarks and never reached
            # the stronger reviewer.
            and decision.get("text_origin") == "scene_text"
            and decision.get("classification") == "sign_or_ui"
            and float(decision.get("confidence") or 0.0) >= keep_confidence
        ):
            decision["decision_source"] = "gemini_lite_parallel_safe_keep"
            safe_keep.append(decision)
    return safe_keep, {
        "enabled": True,
        "status": (
            "completed" if len(results) == len(batches)
            else "partial" if results else "bypassed_after_error"
        ),
        "provider": str(config["parallel_triage_provider"]),
        "model": (
            ANTIGRAVITY_TRIAGE_MODEL
            if config["parallel_triage_provider"] == ANTIGRAVITY_TRIAGE_PROVIDER
            else str(config["parallel_triage_model"])
        ),
        "thinking_level": "minimal",
        "input_event_count": len(events),
        "safe_keep_count": len(safe_keep),
        "forwarded_to_strong_count": len(events) - len(safe_keep),
        "batch_size": batch_size,
        "batch_count": len(batches),
        "completed_batch_count": len(results),
        "failed_batch_count": len(batches) - len(results),
        "worker_count": worker_count,
        "media_scope": "contact_sheets_only",
        "safe_keep_confidence": keep_confidence,
    }


def _strong_review_checkpoint_identity(
    profile: dict[str, Any],
    prompt: str,
    sheets: list[Path],
    config: dict[str, Any],
) -> str:
    # Hash the evidence bytes, not filenames: OCR assets can be regenerated
    # in the same workspace. Only a digest of configuration is persisted.
    sheet_hashes = []
    for sheet in sheets:
        with Path(sheet).open("rb") as stream:
            sheet_hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
    payload = {
        "version": 1,
        "profile": profile,
        "prompt": prompt,
        "sheets": sheet_hashes,
        "config": config,
        "models": list(GEMINI_REVIEW_MODELS),
        "response_schema": gemini_response_schema(),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_strong_review_checkpoint(
    path: Path,
    identity: str,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(cached, dict)
            or cached.get("identity") != identity
            or cached.get("model") not in GEMINI_REVIEW_MODELS
            or not isinstance(cached.get("response"), dict)
        ):
            return None
        response_digest = hashlib.sha256(json.dumps(
            cached["response"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if cached.get("response_sha256") != response_digest:
            return None
        decisions = validate_gemini_decisions(
            profile, cached["response"], config,
        )["decisions"]
        return cached["model"], decisions
    except (OSError, ValueError, TypeError, SilentTextLocalizationError):
        return None


def _save_strong_review_checkpoint(
    path: Path, identity: str, model: str, response: dict[str, Any],
) -> None:
    response_digest = hashlib.sha256(json.dumps(
        response, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "identity": identity,
        "model": model,
        "response_sha256": response_digest,
        "response": response,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _policy_block_fail_closed_response(
    profile: dict[str, Any],
    block_reason: str,
) -> dict[str, Any]:
    """Return schema-valid conservative decisions when every review model blocks."""

    reason = str(block_reason or "UNKNOWN").strip().upper()
    return {
        "items": [
            {
                "track_id": str(event.get("track_id") or ""),
                "classification": "uncertain",
                "text_origin": "uncertain",
                "source_text_corrected": str(event.get("ocr_best_text") or "").strip(),
                "vietnamese_translation": "",
                "highlight_terms": [],
                "design_role": "neutral",
                "replacement_action": "needs_review",
                "confidence": 0.0,
                "reason_vi": (
                    f"Gemini ch?n n?i dung ({reason}); gi? nguy?n ch? ngu?n ?? "
                    "tr?nh x?a ho?c d?ch khi ch?a ???c x?c minh."
                ),
            }
            for event in profile.get("events") or []
            if str(event.get("track_id") or "")
        ]
    }


def _parallel_strong_review(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    clients: list[GeminiRestClient],
    config: dict[str, Any],
    work_folder: str | Path,
    srt_text: str,
    log: Callable[[str], None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Review strong-model batches concurrently without sharing a key."""

    events = list(profile.get("events") or [])
    batches: list[tuple[int, dict[str, Any]]] = []
    for batch_number, start in enumerate(
        range(0, len(events), GEMINI_REVIEW_BATCH_EVENTS),
        start=1,
    ):
        batch_profile = dict(profile)
        batch_profile["events"] = events[
            start : start + GEMINI_REVIEW_BATCH_EVENTS
        ]
        batch_profile["event_count"] = len(batch_profile["events"])
        batches.append((batch_number, batch_profile))
    if not batches:
        return [], {
            "configured_worker_count": int(config["strong_parallel_workers"]),
            "effective_worker_count": 0,
            "rate_limited_key_count": 0,
            "fallback_batch_count": 0,
            "used_models": [],
            "preferred_client_index": 0,
            "uncertain_ids": [],
            "editorial_ui_ids": [],
            "policy_blocked_batch_count": 0,
            "policy_fail_closed_batch_count": 0,
            "policy_blocked_models": [],
        }

    configured_workers = min(
        len(batches),
        len(clients),
        int(config["strong_parallel_workers"]),
    )
    client_locks = [Lock() for _client in clients]
    state_lock = Lock()
    unavailable_by_model: dict[str, set[int]] = {
        model_name: set() for model_name in GEMINI_REVIEW_MODELS
    }
    cooldown_until: dict[str, dict[int, float]] = {
        model_name: {} for model_name in GEMINI_REVIEW_MODELS
    }
    auth_disabled_clients: set[int] = set()
    suspended_models: set[str] = set()
    model_503_key_switches = 0
    permission_disabled_routes: set[tuple[str, int]] = set()
    rate_limited_keys: set[int] = set()
    quota_dimension_counts: dict[str, int] = {}

    def mark_unavailable(model_name: str, client_index: int) -> None:
        with state_lock:
            unavailable_by_model[model_name].add(client_index)

    def mark_auth_disabled(client_index: int) -> None:
        with state_lock:
            auth_disabled_clients.add(client_index)

    def suspend_model(model_name: str) -> None:
        with state_lock:
            suspended_models.add(model_name)

    def is_model_suspended(model_name: str) -> bool:
        with state_lock:
            return model_name in suspended_models

    def is_unavailable(model_name: str, client_index: int) -> bool:
        with state_lock:
            return (
                model_name in suspended_models
                or client_index in auth_disabled_clients
                or client_index in unavailable_by_model[model_name]
                or cooldown_until[model_name].get(client_index, 0.0) > time.monotonic()
            )

    def generate_strong_on_key(
        gemini: GeminiRestClient,
        *,
        model_name: str,
        prompt: str,
        sheets: list[Path],
        batch_profile: dict[str, Any],
        batch_number: int,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        consecutive_503 = 0
        generic_transient_attempt = 0
        while True:
            try:
                decisions: list[dict[str, Any]] = []
                response: dict[str, Any] = {}
                for contract_attempt in range(2):
                    response = gemini.generate(
                        model=model_name,
                        prompt=(
                            prompt
                            if contract_attempt == 0
                            else (
                                prompt
                                + "\n\nLần trả lời trước không vượt qua "
                                "kiểm tra an toàn. Hãy trả lại toàn bộ "
                                "item đúng track_id và đúng mọi quy tắc."
                            )
                        ),
                        media=[],
                        inline_media_paths=sheets,
                        request_timeout_seconds=INLINE_REVIEW_TIMEOUT_SECONDS,
                        request_attempts=1,
                    )
                    try:
                        decisions = validate_gemini_decisions(
                            batch_profile,
                            response,
                            config,
                        )["decisions"]
                        break
                    except SilentTextLocalizationError:
                        if contract_attempt:
                            raise
                        log(
                            "Gemini strong batch failed contract validation; "
                            "retrying once on the same key."
                        )
                return response, decisions
            except GeminiRequestError as exc:
                status = exc.status_code
                if status == 503:
                    # This is intentionally a STRICT consecutive-HTTP-503
                    # counter. A transport timeout or HTTP 429 must never
                    # make two HTTP 503 responses count as three.
                    consecutive_503 += 1
                    generic_transient_attempt = 0
                    if consecutive_503 >= MODEL_503_RETRY_LIMIT:
                        raise
                    delay = float(MODEL_503_BACKOFF_SECONDS[consecutive_503 - 1])
                    log(
                        f"Gemini {model_name} batch {batch_number}/{len(batches)} "
                        f"HTTP 503 on same key {consecutive_503}/{MODEL_503_RETRY_LIMIT}; "
                        f"retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)
                    continue
                consecutive_503 = 0
                if status not in {None, 408, 500, 502, 504}:
                    raise
                generic_transient_attempt += 1
                if generic_transient_attempt >= GENERIC_TRANSIENT_RETRY_LIMIT:
                    raise
                delay = float(MODEL_503_BACKOFF_SECONDS[
                    min(generic_transient_attempt - 1, len(MODEL_503_BACKOFF_SECONDS) - 1)
                ])
                log(
                    f"Gemini {model_name} batch {batch_number}/{len(batches)} "
                    f"HTTP {status or 'transport'} (NOT a 503 key-cycle event); "
                    f"retrying the same key in {delay:.1f}s."
                )
                time.sleep(delay)

    def review_one(
        batch_number: int,
        batch_profile: dict[str, Any],
        allow_cooldown: bool = True,
        start_client_override: int | None = None,
        completed_503_cycles: int = 0,
    ) -> dict[str, Any]:
        nonlocal model_503_key_switches
        sheets = build_contact_sheets(
            batch_profile,
            profile_path,
            Path(work_folder)
            / "contact-sheets"
            / f"batch-{batch_number:03d}",
        )
        prompt = build_gemini_prompt(batch_profile, srt_text=srt_text)
        checkpoint_path = (
            Path(work_folder) / "strong-review-checkpoints"
            / f"batch-{batch_number:03d}.json"
        )
        identity = _strong_review_checkpoint_identity(
            batch_profile, prompt, sheets, config,
        )
        cached = _load_strong_review_checkpoint(
            checkpoint_path, identity, batch_profile, config,
        )
        if cached is not None:
            cached_model, cached_decisions = cached
            log(f"Reusing verified strong review batch {batch_number}/{len(batches)}.")
            return {
                "batch_number": batch_number,
                "decisions": cached_decisions,
                "model": cached_model,
                "client_index": (batch_number - 1) % len(clients),
                "uncertain_ids": {
                    str(item.get("track_id") or "")
                    for item in cached_decisions
                    if item.get("classification") == "uncertain"
                    or item.get("text_origin") == "uncertain"
                    or item.get("replacement_action") == "needs_review"
                },
                "editorial_ui_ids": _editorial_ui_second_review_ids(
                    batch_profile, cached_decisions,
                ),
                "checkpoint_reused": True,
                "policy_blocked": False,
                "policy_fail_closed": False,
                "policy_blocked_models": [],
            }
        last_error: GeminiRequestError | None = None
        policy_blocked_models: list[str] = []
        last_policy_block_reason = ""
        start_client = (
            (batch_number - 1) % len(clients)
            if start_client_override is None else start_client_override
        )
        fallback_client: int | None = None
        next_503_client: int | None = None
        for model_name in GEMINI_REVIEW_MODELS:
            if (start_client_override is not None
                    and model_name == GEMINI_REVIEW_MODELS[0]
                    and is_unavailable(model_name, start_client)):
                fallback_client = start_client
                continue
            model_policy_blocked = False
            for offset in range(len(clients)):
                model_start = fallback_client if fallback_client is not None else start_client
                client_index = (model_start + offset) % len(clients)
                if is_unavailable(model_name, client_index):
                    continue
                gemini = clients[client_index]
                with client_locks[client_index]:
                    if is_unavailable(model_name, client_index):
                        continue
                    try:
                        response, decisions = generate_strong_on_key(
                            gemini,
                            model_name=model_name,
                            prompt=prompt,
                            sheets=sheets,
                            batch_profile=batch_profile,
                            batch_number=batch_number,
                        )
                    except GeminiPolicyBlockError as exc:
                        model_policy_blocked = True
                        last_policy_block_reason = exc.block_reason
                        policy_blocked_models.append(model_name)
                        log(
                            f"Gemini strong batch {batch_number}/{len(batches)} "
                            f"policy-blocked by {model_name} ({exc.block_reason}); "
                            "not rotating keys for identical content."
                        )
                        break
                    except GeminiRequestError as exc:
                        last_error = exc
                        status_code = exc.status_code
                        status = (
                            f"HTTP {status_code}"
                            if status_code is not None
                            else "transport error"
                        )
                        if status_code == 503:
                            if model_name == GEMINI_REVIEW_MODELS[-1]:
                                next_503_client = (client_index + 1) % len(clients)
                            else:
                                fallback_client = client_index
                            log(
                                f"Gemini {model_name} returned 503 after "
                                f"{MODEL_503_RETRY_LIMIT} same-key attempts on key "
                                f"{client_index + 1}/{len(clients)}; trying the next model/key."
                            )
                            break
                        if status_code == 404:
                            suspend_model(model_name)
                            log(
                                f"Gemini {model_name} is unavailable (HTTP 404); "
                                "falling back by model without rotating keys."
                            )
                            break
                        if status_code == 401:
                            mark_auth_disabled(client_index)
                            log(
                                f"Gemini key {client_index + 1}/{len(clients)} failed authentication; "
                                "disabling that key for all models in this run."
                            )
                            continue
                        if status_code == 403:
                            mark_unavailable(model_name, client_index)
                            with state_lock:
                                permission_disabled_routes.add((model_name, client_index))
                            log(
                                f"Gemini {model_name} key {client_index + 1}/{len(clients)} "
                                "has no permission; disabling only this key/model route."
                            )
                            continue
                        if status_code == 429:
                            dimension = exc.quota_dimension or _quota_dimension(str(exc))
                            with state_lock:
                                rate_limited_keys.add(client_index)
                                quota_dimension_counts[dimension] = (
                                    quota_dimension_counts.get(dimension, 0) + 1
                                )
                                if dimension == "RPD":
                                    unavailable_by_model[model_name].add(client_index)
                                else:
                                    delay = exc.retry_after_seconds
                                    if delay is None:
                                        delay = (
                                            UNKNOWN_429_COOLDOWN_SECONDS
                                            if dimension == "UNKNOWN"
                                            else QUOTA_429_COOLDOWN_SECONDS
                                        )
                                    cooldown_until[model_name][client_index] = max(
                                        cooldown_until[model_name].get(client_index, 0.0),
                                        time.monotonic() + max(0.0, float(delay)),
                                    )
                            log(
                                f"Gemini {model_name} key {client_index + 1}/{len(clients)} "
                                f"hit HTTP 429 {dimension}; rotating to another project while this route "
                                + ("stays exhausted for this run." if dimension == "RPD" else "cools down.")
                            )
                            continue
                        if not exc.rotate_key:
                            raise
                        with state_lock:
                            cooldown_until[model_name][client_index] = max(
                                cooldown_until[model_name].get(client_index, 0.0),
                                time.monotonic() + UNKNOWN_429_COOLDOWN_SECONDS,
                            )
                        log(
                            f"Gemini {model_name} key {client_index + 1}/{len(clients)} "
                            f"unavailable after bounded same-key retry ({status}); rotating once."
                        )
                        continue
                if not decisions:
                    raise SilentTextLocalizationError(
                        "Gemini strong batch produced no validated decisions."
                    )
                _save_strong_review_checkpoint(
                    checkpoint_path, identity, model_name, response,
                )
                uncertain_ids = {
                    str(item.get("track_id") or "")
                    for item in decisions
                    if item.get("classification") == "uncertain"
                    or item.get("text_origin") == "uncertain"
                    or item.get("replacement_action") == "needs_review"
                }
                editorial_ids = _editorial_ui_second_review_ids(
                    batch_profile,
                    decisions,
                )
                return {
                    "batch_number": batch_number,
                    "decisions": decisions,
                    "model": model_name,
                    "client_index": client_index,
                    "uncertain_ids": uncertain_ids,
                    "editorial_ui_ids": editorial_ids,
                    "checkpoint_reused": False,
                    "policy_blocked": bool(policy_blocked_models),
                    "policy_fail_closed": False,
                    "policy_blocked_models": list(policy_blocked_models),
                }
            if model_policy_blocked:
                if model_name != GEMINI_REVIEW_MODELS[-1]:
                    next_model = GEMINI_REVIEW_MODELS[
                        GEMINI_REVIEW_MODELS.index(model_name) + 1
                    ]
                    log(
                        f"Policy block on {model_name}; falling back to {next_model} "
                        f"for strong batch {batch_number}/{len(batches)}."
                    )
                continue
            with state_lock:
                now = time.monotonic()
                cooldown_seconds = min(
                    (
                        deadline - now
                        for key_index, deadline in cooldown_until[model_name].items()
                        if key_index not in unavailable_by_model[model_name]
                        and key_index not in auth_disabled_clients
                        and deadline > now
                    ),
                    default=0.0,
                )
                model_suspended = model_name in suspended_models
            if (allow_cooldown and fallback_client is None
                    and next_503_client is None
                    and not model_suspended and cooldown_seconds > 0):
                log(
                    f"No {model_name} key is currently usable; waiting "
                    f"{cooldown_seconds:.1f}s, then retrying the highest model "
                    "before any fallback."
                )
                time.sleep(cooldown_seconds)
                return review_one(
                    batch_number,
                    batch_profile,
                    allow_cooldown=False,
                    start_client_override=start_client_override,
                    completed_503_cycles=completed_503_cycles,
                )
            if model_name != GEMINI_REVIEW_MODELS[-1]:
                next_model = GEMINI_REVIEW_MODELS[
                    GEMINI_REVIEW_MODELS.index(model_name) + 1
                ]
                log(
                    f"All currently usable {model_name} keys are exhausted; "
                    f"falling back to {next_model} for this batch."
                )
        if next_503_client is not None:
            completed_503_cycles += 1
            if completed_503_cycles < min(MAX_503_KEY_CYCLES, len(clients)):
                with state_lock:
                    model_503_key_switches += 1
                log(
                    f"Both strong-review models unavailable with HTTP 503; "
                    f"starting key cycle {completed_503_cycles + 1}/"
                    f"{min(MAX_503_KEY_CYCLES, len(clients))} at key "
                    f"{next_503_client + 1}/{len(clients)}."
                )
                return review_one(
                    batch_number,
                    batch_profile,
                    allow_cooldown=allow_cooldown,
                    start_client_override=next_503_client,
                    completed_503_cycles=completed_503_cycles,
                )
        if policy_blocked_models:
            response = _policy_block_fail_closed_response(
                batch_profile,
                last_policy_block_reason,
            )
            decisions = validate_gemini_decisions(
                batch_profile,
                response,
                config,
            )["decisions"]
            log(
                f"All strong review models policy-blocked batch "
                f"{batch_number}/{len(batches)}; preserving {len(decisions)} "
                "event(s) with local fail-closed restore decisions."
            )
            return {
                "batch_number": batch_number,
                "decisions": decisions,
                "model": policy_blocked_models[-1],
                "client_index": start_client,
                "uncertain_ids": {
                    str(item.get("track_id") or "") for item in decisions
                },
                "editorial_ui_ids": set(),
                "checkpoint_reused": False,
                "policy_blocked": True,
                "policy_fail_closed": True,
                "policy_blocked_models": list(policy_blocked_models),
            }
        status = (
            f"HTTP {last_error.status_code}"
            if last_error and last_error.status_code is not None
            else "transport error"
        )
        raise SilentTextLocalizationError(
            "All configured Gemini keys/models are unavailable for batch "
            f"{batch_number}/{len(batches)} ({status})."
        )

    results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=configured_workers) as executor:
        futures = {
            executor.submit(review_one, batch_number, batch_profile): batch_number
            for batch_number, batch_profile in batches
        }
        for future in as_completed(futures):
            result = future.result()
            results[int(result["batch_number"])] = result
            log(
                "Gemini strong review batch completed: "
                f"{result['batch_number']}/{len(batches)}; "
                f"events={len(result['decisions'])}; model={result['model']}."
            )

    ordered = [results[index] for index in range(1, len(batches) + 1)]
    return [
        decision
        for result in ordered
        for decision in result["decisions"]
    ], {
        "configured_worker_count": configured_workers,
        "checkpoint_reused_batch_count": sum(
            bool(result["checkpoint_reused"]) for result in ordered
        ),
        "effective_worker_count": configured_workers,
        "rate_limited_key_count": len(rate_limited_keys),
        "quota_dimension_counts": dict(sorted(quota_dimension_counts.items())),
        "auth_disabled_key_count": len(auth_disabled_clients),
        "permission_disabled_route_count": len(permission_disabled_routes),
        "model_503_suspend_count": 0,
        "model_503_key_switch_count": model_503_key_switches,
        "fallback_batch_count": sum(
            result["model"] != GEMINI_REVIEW_MODELS[0] for result in ordered
        ),
        "used_models": [result["model"] for result in ordered],
        "preferred_client_index": int(ordered[-1]["client_index"]),
        "uncertain_ids": sorted(
            {
                track_id
                for result in ordered
                for track_id in result["uncertain_ids"]
            }
        ),
        "editorial_ui_ids": sorted(
            {
                track_id
                for result in ordered
                for track_id in result["editorial_ui_ids"]
            }
        ),
        "policy_blocked_batch_count": sum(
            bool(result.get("policy_blocked")) for result in ordered
        ),
        "policy_fail_closed_batch_count": sum(
            bool(result.get("policy_fail_closed")) for result in ordered
        ),
        "policy_blocked_models": list(dict.fromkeys(
            model_name
            for result in ordered
            for model_name in result.get("policy_blocked_models") or []
        )),
    }


def request_gemini_review(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    video_path: str | Path,
    config: object,
    work_folder: str | Path,
    srt_path: str | Path | None = None,
    client: GeminiRestClient | None = None,
    api_keys: list[str] | tuple[str, ...] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    lite_triage_explicitly_configured = bool(
        isinstance(config, dict)
        and config.get("lite_parallel_triage_enabled") is True
    )
    normalized_config = normalize_localization_config(config)
    if not normalized_config["enabled"]:
        raise SilentTextLocalizationError("Silent-text localization is disabled.")
    logger = log or (lambda _message: None)
    logger("Gemini review flow: semantic-batch-v1.")
    source_video = Path(video_path)
    if not source_video.is_file():
        raise SilentTextLocalizationError(f"Video is missing: {source_video}")
    srt = Path(srt_path) if srt_path else discover_srt(source_video)
    srt_text = srt.read_text(encoding="utf-8-sig") if srt and srt.is_file() else ""
    align_subtitle_events_with_srt(profile, srt_text)
    speech_alignment = profile.get("speech_alignment") or {}
    logger(
        "Local OCR-SRT alignment: "
        f"status={speech_alignment.get('status')}; "
        f"cues={speech_alignment.get('cue_count', 0)}; "
        f"matched={speech_alignment.get('matched_count', 0)}; "
        f"ambiguous={speech_alignment.get('ambiguous_count', 0)}; "
        f"unmatched={speech_alignment.get('unmatched_count', 0)}; "
        f"offset_ms={speech_alignment.get('timeline_offset_ms', 0)}."
    )
    if client is not None:
        clients = [client]
    elif api_keys:
        clients = [
            GeminiRestClient(str(api_key), log=logger)
            for api_key in api_keys
        ]
    else:
        clients = [
            GeminiRestClient(api_key, log=logger)
            for _name, api_key in gemini_api_key_pool(normalized_config)
        ]
    triage_clients: list[GeminiRestClient] = clients
    if (
        normalized_config["parallel_triage_provider"]
        == ANTIGRAVITY_TRIAGE_PROVIDER
        and client is None
    ):
        if api_keys:
            triage_clients = [
                AntigravityRestClient(str(api_key), log=logger)
                for api_key in api_keys
            ]
        else:
            triage_clients = [
                AntigravityRestClient(api_key, log=logger)
                for _name, api_key in gemini_api_key_pool(normalized_config)
            ]
        logger(
            "Antigravity parallel triage enabled: bounded contact-sheet "
            "interactions; Gemini remains the strong-review path."
        )
    local_spoken_events = _trusted_local_spoken_events(profile)
    speech_review_routing = profile.get("speech_review_routing") or {}
    local_spoken_ids = {
        str(event.get("track_id") or "") for event in local_spoken_events
    }
    local_middle_microtext_events = [
        event
        for event in profile.get("events") or []
        if str(event.get("track_id") or "") not in local_spoken_ids
        and _is_short_small_middle_event(event, normalized_config)
    ]
    local_middle_microtext_ids = {
        str(event.get("track_id") or "")
        for event in local_middle_microtext_events
    }
    local_weak_ocr_fragment_events = [
        event
        for event in profile.get("events") or []
        if str(event.get("track_id") or "") not in local_spoken_ids
        and str(event.get("track_id") or "") not in local_middle_microtext_ids
        and _is_weak_primary_review_ocr_fragment(event)
    ]
    local_weak_ocr_fragment_ids = {
        str(event.get("track_id") or "")
        for event in local_weak_ocr_fragment_events
    }
    gemini_input_profile = dict(profile)
    gemini_input_profile["events"] = [
        event
        for event in profile.get("events") or []
        if str(event.get("track_id") or "") not in local_spoken_ids
        and str(event.get("track_id") or "") not in local_middle_microtext_ids
        and str(event.get("track_id") or "") not in local_weak_ocr_fragment_ids
    ]
    gemini_input_profile["event_count"] = len(gemini_input_profile["events"])
    logger(
        "Local spoken-subtitle routing: "
        f"removed_before_gemini={len(local_spoken_events)}; "
        f"gemini_input_events={gemini_input_profile['event_count']}; "
        f"minimum_score={LOCAL_SPOKEN_ROUTE_MIN_SCORE:.2f}; "
        f"max_duration_ratio={LOCAL_SPOKEN_ROUTE_MAX_DURATION_RATIO:.1f}; "
        "temporal_shape_rejected="
        f"{speech_review_routing.get('rejected_temporal_shape_count', 0)}; "
        "text_shape_rejected="
        f"{speech_review_routing.get('rejected_text_shape_count', 0)}."
    )
    logger(
        "Local short-small-middle routing: "
        f"kept_before_gemini={len(local_middle_microtext_events)}; "
        f"max_duration={normalized_config['middle_microtext_max_duration_seconds']:.2f}s; "
        f"max_area={normalized_config['middle_microtext_max_area_fraction']:.4f}; "
        f"max_height={normalized_config['middle_microtext_max_height_fraction']:.4f}; "
        f"middle_band={normalized_config['middle_microtext_top_fraction']:.2f}-"
        f"{normalized_config['middle_microtext_bottom_fraction']:.2f}."
    )
    logger(
        "Local weak-OCR fragment routing: "
        f"deferred_before_gemini={len(local_weak_ocr_fragment_events)}; "
        f"gemini_input_events={gemini_input_profile['event_count']}; "
        f"max_duration={PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_DURATION_SECONDS:.2f}s; "
        f"max_observed_frames={PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_OBSERVED_FRAMES}."
    )
    if normalized_config["scenery_guard_enabled"]:
        guard_results, guard_models = _request_scenery_guard(
            profile=gemini_input_profile,
            profile_path=profile_path,
            clients=clients,
            log=logger,
        )
    else:
        guard_results, guard_models = {}, []
        logger(
            "Gemini Flash Lite scenery guard skipped: the strong semantic "
            "review is authoritative and the advisory guard does not remove "
            "events from its input."
        )
    review_profile = dict(gemini_input_profile)
    review_profile["events"] = [
        {
            **event,
            "scenery_guard_hint": guard_results.get(
                str(event.get("track_id") or "")
            ),
        }
        for event in gemini_input_profile.get("events") or []
    ]
    review_profile["event_count"] = len(review_profile["events"])
    lite_not_text_ids = {
        track_id
        for track_id, result in guard_results.items()
        if result.get("verdict") == "NOT_TEXT"
        and float(result.get("confidence") or 0.0)
        >= SCENERY_NOT_TEXT_CONFIDENCE
    }
    if lite_not_text_ids:
        logger(
            "Gemini Flash Lite scenery guard flagged "
            f"{len(lite_not_text_ids)} possible OCR false positives; forwarding "
            "all of them to the strong Gemini review for confirmation."
        )

    semantic_profile, children_by_semantic_id = _semantic_review_profile(review_profile)
    semantic_count = len(semantic_profile.get("events") or [])
    consolidated_semantic_count = semantic_count
    exact_count = len(review_profile.get("events") or [])
    if exact_count:
        logger(
            "Gemini semantic consolidation: "
            f"exact_events={exact_count}; semantic_events={semantic_count}; "
            f"reduction={exact_count - semantic_count}."
        )
    fragmentation_guard = _semantic_fragmentation_guard_metrics(
        review_profile,
        exact_count=exact_count,
        semantic_count=semantic_count,
    )
    logger(
        "Semantic fragmentation health: "
        f"retention={fragmentation_guard['retention_ratio']:.4f}; "
        f"events_per_second={fragmentation_guard['events_per_second']:.4f}; "
        f"strong_batches={fragmentation_guard['batch_count']}."
    )
    if fragmentation_guard["blocked"]:
        raise SilentTextLocalizationError(
            "Semantic event graph is too fragmented for safe strong review: "
            f"events={semantic_count}/{exact_count}; "
            f"density={fragmentation_guard['events_per_second']:.2f}/s; "
            f"batches={fragmentation_guard['batch_count']}. "
            "Refusing to spend Gemini calls until local consolidation improves."
        )

    lite_triage = {
        "enabled": False,
        "input_event_count": consolidated_semantic_count,
        "safe_keep_count": 0,
        "forwarded_to_strong_count": consolidated_semantic_count,
        "batch_count": 0,
        "worker_count": 0,
    }
    lite_safe_decisions: list[dict[str, Any]] = []
    if (
        normalized_config["lite_parallel_triage_enabled"]
        and lite_triage_explicitly_configured
        and semantic_count
    ):
        try:
            lite_safe_decisions, lite_triage = _parallel_lite_triage(
                profile=semantic_profile,
                profile_path=profile_path,
                source_video=source_video,
                clients=triage_clients,
                config=normalized_config,
                work_folder=work_folder,
                srt_text=srt_text,
                log=logger,
            )
        except SilentTextLocalizationError as exc:
            lite_safe_decisions = []
            lite_triage = {
                "enabled": True,
                "status": "bypassed_after_error",
                "error_type": type(exc).__name__,
                "provider": normalized_config["parallel_triage_provider"],
                "model": (
                    ANTIGRAVITY_TRIAGE_MODEL
                    if normalized_config["parallel_triage_provider"]
                    == ANTIGRAVITY_TRIAGE_PROVIDER
                    else normalized_config["parallel_triage_model"]
                ),
                "input_event_count": semantic_count,
                "safe_keep_count": 0,
                "forwarded_to_strong_count": semantic_count,
                "batch_count": math.ceil(
                    semantic_count
                    / int(normalized_config["lite_parallel_batch_size"])
                ),
                "worker_count": min(
                    len(clients),
                    int(normalized_config["lite_parallel_workers"]),
                ),
            }
            logger(
                "Gemini Lite triage unavailable; forwarding every semantic "
                f"event to the strong review ({type(exc).__name__})."
            )
        safe_keep_ids = {
            str(item.get("track_id") or "") for item in lite_safe_decisions
        }
        semantic_profile = {
            **semantic_profile,
            "events": [
                event
                for event in semantic_profile.get("events") or []
                if str(event.get("track_id") or "") not in safe_keep_ids
            ],
        }
        semantic_profile["event_count"] = len(semantic_profile["events"])
        semantic_count = semantic_profile["event_count"]
        logger(
            "Gemini Lite triage completed before strong review: "
            f"safe_keep={len(lite_safe_decisions)}; "
            f"forwarded_to_strong={semantic_count}; "
            f"workers={lite_triage['worker_count']}."
        )

    semantic_decisions: list[dict[str, Any]] = list(lite_safe_decisions)
    used_review_models: list[str] = (
        [str(normalized_config["parallel_triage_model"])]
        if lite_safe_decisions
        else []
    )
    editorial_ui_second_review_ids: set[str] = set()
    uncertain_second_review_ids: set[str] = set()
    focused_ocr_guard_rejected_ids: set[str] = set()
    focused_ocr_pre_gate_skipped_ids: set[str] = set()
    focused_second_review_batch_count = 0
    preferred_client_index = 0
    strong_parallel = {
        "enabled": False,
        "configured_worker_count": 0,
        "effective_worker_count": 0,
        "rate_limited_key_count": 0,
        "fallback_batch_count": 0,
    }
    if semantic_profile["events"]:
        batch_count = math.ceil(
            len(semantic_profile["events"]) / GEMINI_REVIEW_BATCH_EVENTS
        )
        logger(
            "Gemini strong review batching: "
            f"events={semantic_count}; batch_size={GEMINI_REVIEW_BATCH_EVENTS}; "
            f"batches={batch_count}."
        )
        primary_decisions, parallel_meta = _parallel_strong_review(
            profile=semantic_profile,
            profile_path=profile_path,
            clients=clients,
            config=normalized_config,
            work_folder=work_folder,
            srt_text=srt_text,
            log=logger,
        )
        semantic_decisions.extend(primary_decisions)
        used_review_models.extend(parallel_meta["used_models"])
        preferred_client_index = int(parallel_meta["preferred_client_index"])
        uncertain_second_review_ids.update(parallel_meta["uncertain_ids"])
        editorial_ui_second_review_ids.update(
            parallel_meta["editorial_ui_ids"]
        )
        strong_parallel = {"enabled": True, **parallel_meta}
        repaired_multiline_models = _repair_persistent_multiline_bridges(
            profile=semantic_profile,
            profile_path=profile_path,
            decisions=semantic_decisions,
            clients=clients,
            config=normalized_config,
            work_folder=work_folder,
            srt_text=srt_text,
            preferred_client_index=preferred_client_index,
            log=logger,
        )
        if repaired_multiline_models:
            used_review_models.extend(repaired_multiline_models)
        try:
            event_by_id = {
                str(event.get("track_id") or ""): event
                for event in semantic_profile.get("events") or []
            }
            focused_ocr_pre_gate_skipped_ids = {
                track_id
                for track_id in uncertain_second_review_ids
                if track_id not in editorial_ui_second_review_ids
                and not _focused_second_review_has_local_ocr_evidence(
                    event_by_id.get(track_id) or {}
                )
            }
            if focused_ocr_pre_gate_skipped_ids:
                logger(
                    "Focused Gemini OCR pre-gate kept fail-closed decisions locally: "
                    f"skipped={len(focused_ocr_pre_gate_skipped_ids)}; "
                    "reason=no high-confidence local OCR evidence."
                )
            focused_ids = (
                uncertain_second_review_ids | editorial_ui_second_review_ids
            ) - focused_ocr_pre_gate_skipped_ids
            primary_decision_by_id = {
                str(item.get("track_id") or ""): item
                for item in semantic_decisions
            }
            focused_events = [
                event
                for event in semantic_profile.get("events") or []
                if str(event.get("track_id") or "") in focused_ids
            ]
            exact_product_profile = all(
                isinstance(event.get("frame_boxes"), list)
                for event in semantic_profile.get("events") or []
            )
            if (
                focused_events
                and exact_product_profile
                and normalized_config["focused_second_review_enabled"]
            ):
                focused_batch_count = math.ceil(
                    len(focused_events) / GEMINI_REVIEW_BATCH_EVENTS
                )
                logger(
                    "Gemini focused second review batching: "
                    f"events={len(focused_events)}; "
                    f"batch_size={GEMINI_REVIEW_BATCH_EVENTS}; "
                    f"batches={focused_batch_count}."
                )
                replacement_by_id: dict[str, dict[str, Any]] = {}
                for focused_batch_index, start in enumerate(
                    range(0, len(focused_events), GEMINI_REVIEW_BATCH_EVENTS),
                    start=1,
                ):
                    focused_profile = dict(semantic_profile)
                    focused_profile["events"] = focused_events[
                        start : start + GEMINI_REVIEW_BATCH_EVENTS
                    ]
                    focused_profile["event_count"] = len(
                        focused_profile["events"]
                    )
                    focused_sheets = build_contact_sheets(
                        focused_profile,
                        profile_path,
                        Path(work_folder)
                        / "contact-sheets"
                        / f"focused-{focused_batch_index:03d}",
                    )
                    focused_prompt = (
                        build_gemini_prompt(
                            focused_profile,
                            srt_text=srt_text,
                        )
                        + "\n\nĐây là lượt xem lại tập trung cho các track chưa "
                        "chắc chắn hoặc có nguy cơ nhầm câu chú thích biên tập thành UI/"
                        "watermark. "
                        "This is a focused second review for tracks that "
                        "were uncertain or may have been mistaken for UI or a "
                        "watermark. Compare positions across the representative "
                        "images, scene context, and text function carefully. No "
                        "continuous video or audio is attached. Flat editorial "
                        "disclaimers, warnings, "
                        "informational captions, and narrative overlays remain "
                        "post-production text. Large readable channel or series "
                        "names inside a designed editorial card must also be "
                        "localized as narrative text. Only real platform UI, "
                        "small source-attribution handles/logos, and text "
                        "physically in the scene must remain untouched. "
                        "If evidence is insufficient, return uncertain."
                    )
                    focused_completed = False
                    focused_client_order = [
                        *range(preferred_client_index, len(clients)),
                        *range(0, preferred_client_index),
                    ]
                    for client_index in focused_client_order:
                        gemini = clients[client_index]
                        try:
                            for attempt in range(2):
                                response = gemini.generate(
                                    model=PRIMARY_GEMINI_MODEL,
                                    prompt=(
                                        focused_prompt
                                        if attempt == 0
                                        else (
                                            focused_prompt
                                            + "\n\nThe previous response failed the "
                                            "contract. Return every item again with "
                                            "the exact track_id and required fields."
                                        )
                                    ),
                                    media=[],
                                    inline_media_paths=focused_sheets,
                                    request_timeout_seconds=(
                                        INLINE_REVIEW_TIMEOUT_SECONDS
                                    ),
                                    request_attempts=MODEL_503_RETRY_LIMIT,
                                )
                                try:
                                    focused_decisions = validate_gemini_decisions(
                                        focused_profile,
                                        response,
                                        normalized_config,
                                    )["decisions"]
                                except SilentTextLocalizationError:
                                    if attempt:
                                        raise
                                    logger(
                                        "Focused Gemini batch failed contract "
                                        "validation; retrying once."
                                    )
                                else:
                                    focused_event_by_id = {
                                        str(event.get("track_id") or ""): event
                                        for event in focused_profile.get("events") or []
                                    }
                                    guarded_focused_decisions = []
                                    for item in focused_decisions:
                                        track_id = str(item.get("track_id") or "")
                                        primary = primary_decision_by_id.get(track_id) or {}
                                        event = focused_event_by_id.get(track_id) or {}
                                        if not _focused_second_review_upgrade_allowed(
                                            primary,
                                            item,
                                            event,
                                        ):
                                            focused_ocr_guard_rejected_ids.add(track_id)
                                            logger(
                                                "Focused Gemini replacement upgrade rejected "
                                                f"for {track_id}: no matching high-confidence "
                                                "local OCR evidence; preserving the primary "
                                                "fail-closed decision."
                                            )
                                            continue
                                        guarded_focused_decisions.append(item)
                                    focused_decisions = guarded_focused_decisions
                                    replacement_by_id.update(
                                        {
                                            str(item.get("track_id") or ""): item
                                            for item in focused_decisions
                                        }
                                    )
                                    preferred_client_index = client_index
                                    used_review_models.append(
                                        PRIMARY_GEMINI_MODEL
                                    )
                                    focused_second_review_batch_count += 1
                                    focused_completed = True
                                    logger(
                                        "Gemini focused second review batch completed: "
                                        f"{focused_batch_index}/{focused_batch_count}; "
                                        f"events={len(focused_decisions)}."
                                    )
                                    break
                        except GeminiRequestError as exc:
                            if exc.status_code in {404, 503}:
                                logger(
                                    "Gemini focused review model is temporarily unavailable "
                                    f"(HTTP {exc.status_code}); preserving primary fail-closed "
                                    "decisions without scanning the remaining keys."
                                )
                                break
                            if not exc.rotate_key:
                                raise
                            status = (
                                f"HTTP {exc.status_code}"
                                if exc.status_code is not None
                                else "transport error"
                            )
                            logger(
                                f"Gemini focused review key "
                                f"{client_index + 1}/{len(clients)} unavailable "
                                f"({status}); rotating."
                            )
                        except SilentTextLocalizationError:
                            logger(
                                "Focused Gemini batch did not pass the contract; "
                                "preserving its primary fail-closed decisions."
                            )
                            break
                        if focused_completed:
                            break
                    if not focused_completed:
                        logger(
                            "Focused Gemini batch was not completed; preserving "
                            "its primary fail-closed decisions."
                        )
                if replacement_by_id:
                    semantic_decisions = [
                        replacement_by_id.get(
                            str(item.get("track_id") or ""),
                            item,
                        )
                        for item in semantic_decisions
                    ]
        finally:
            # Inline contact sheets require no remote file cleanup.
            pass

    reviewed_decisions = _expand_semantic_decisions(
        original_profile=gemini_input_profile,
        semantic_decisions=semantic_decisions,
        children_by_semantic_id=children_by_semantic_id,
        config=normalized_config,
    )
    unique_review_models = list(dict.fromkeys(used_review_models))
    used_review_model = ""
    if unique_review_models:
        used_review_model = (
            unique_review_models[0]
            if len(unique_review_models) == 1
            else "mixed"
        )

    local_spoken_decisions = [
        _local_spoken_decision(event) for event in local_spoken_events
    ]
    local_middle_microtext_decisions = [
        _local_middle_microtext_keep_decision(event)
        for event in local_middle_microtext_events
    ]
    local_weak_ocr_fragment_decisions = [
        _local_weak_ocr_fragment_keep_decision(event)
        for event in local_weak_ocr_fragment_events
    ]
    decisions = [
        *local_spoken_decisions,
        *local_middle_microtext_decisions,
        *local_weak_ocr_fragment_decisions,
        *reviewed_decisions,
    ]
    linked_fragment_count = _link_short_text_fragments(profile, decisions)
    if linked_fragment_count:
        logger(
            "Exact track routing attached "
            f"{linked_fragment_count} short OCR fragments to confirmed text tracks."
        )
    promoted_persistent_group_count = _promote_reviewed_persistent_edge_groups(
        profile,
        decisions,
    )
    if promoted_persistent_group_count:
        logger(
            "Reviewed persistent edge promotion formed "
            f"{promoted_persistent_group_count} stable microtext group(s)."
        )
    persistent_envelope_link_count = _link_persistent_overlay_envelopes(
        profile,
        decisions,
    )
    if persistent_envelope_link_count:
        logger(
            "Persistent overlay envelope routing attached "
            f"{persistent_envelope_link_count} stable multi-line track(s) "
            "to reviewed post-production anchors."
        )
    persistent_fragment_inherit_count = _inherit_persistent_edge_fragments(
        profile,
        decisions,
    )
    if persistent_fragment_inherit_count:
        logger(
            "Persistent edge inheritance attached "
            f"{persistent_fragment_inherit_count} small edge fragment(s) "
            "to confirmed post-production anchors."
        )
    suppressed_persistent_nested_render_count = _suppress_persistent_edge_nested_renders(
        profile,
        decisions,
    )
    if suppressed_persistent_nested_render_count:
        logger(
            "Persistent edge render guard suppressed "
            f"{suppressed_persistent_nested_render_count} conflicting nested render(s)."
        )
    persistent_tail_extended_count = _extend_confirmed_persistent_supertrack_tail(
        profile,
        decisions,
    )
    if persistent_tail_extended_count:
        logger(
            "Persistent super-track tail extension reached the processing end for "
            f"{persistent_tail_extended_count} confirmed editor overlay(s)."
        )
    nested_source_fragment_suppressed_count = _suppress_nested_source_fragment_renders(
        decisions
    )
    if nested_source_fragment_suppressed_count:
        logger(
            "Nested source-fragment render guard suppressed "
            f"{nested_source_fragment_suppressed_count} duplicate render(s)."
        )
    translation_consistency_override_count = _canonicalize_repeated_overlay_translations(
        decisions
    )
    if translation_consistency_override_count:
        logger(
            "Vietnamese translation consistency normalized "
            f"{translation_consistency_override_count} repeated editor overlay occurrence(s)."
        )
    multiline_group_count = _reconcile_persistent_multiline_fragments(profile, decisions)
    if multiline_group_count:
        logger(
            "Persistent multiline OCR routing retained "
            f"{multiline_group_count} separately placed reviewed row group(s)."
        )
    suppressed_duplicate_render_count = _deduplicate_vietnamese_render_fragments(
        decisions
    )
    if suppressed_duplicate_render_count:
        logger(
            "Vietnamese render deduplication kept one semantic anchor and "
            f"suppressed {suppressed_duplicate_render_count} duplicate fragment(s)."
        )
    render_overlap_conflicts = _conflicting_vietnamese_render_pairs(decisions)
    if render_overlap_conflicts:
        preview = ", ".join(
            f"{item['first_track_id']}<->{item['second_track_id']}"
            for item in render_overlap_conflicts[:6]
        )
        raise SilentTextLocalizationError(
            "Conflicting Vietnamese render anchors overlap in time and space: "
            f"count={len(render_overlap_conflicts)}; {preview}."
        )
    decisions.sort(
        key=lambda item: (
            int(item.get("start_frame") or 0),
            int(item.get("end_frame") or 0),
            str(item.get("track_id") or ""),
        )
    )
    for decision in decisions:
        guard = guard_results.get(str(decision.get("track_id") or ""))
        if guard and "scenery_guard" not in decision:
            decision["scenery_guard"] = guard
    main_model_preserved_ids = {
        str(decision.get("track_id") or "")
        for decision in decisions
        if str(decision.get("track_id") or "") in lite_not_text_ids
        and not bool(decision.get("selected_for_inpaint"))
        and decision.get("render_action") == "restore_original"
    }
    main_model_selected_ids = {
        str(decision.get("track_id") or "")
        for decision in decisions
        if str(decision.get("track_id") or "") in lite_not_text_ids
        and bool(decision.get("selected_for_inpaint"))
    }
    return {
        "schema": REVIEW_SCHEMA,
        "version": REVIEW_VERSION,
        "source_video_name": profile.get("source_video_name") or "",
        "config": normalized_config,
        "summary": _review_summary(decisions),
        "decisions": decisions,
        "ai_provider": "google-gemini-api",
        "ai_model": used_review_model or None,
        "ai_models": list(dict.fromkeys([*guard_models, *unique_review_models])),
        "evidence_scope": _gemini_evidence_scope(srt_text=srt_text),
        "semantic_consolidation": {
            "exact_event_count": exact_count,
            "semantic_event_count": consolidated_semantic_count,
            "reduction_count": exact_count - consolidated_semantic_count,
        },
        "persistent_overlay_envelopes": {
            "promoted_reviewed_group_count": promoted_persistent_group_count,
            "linked_track_count": persistent_envelope_link_count,
            "inherited_edge_fragment_count": persistent_fragment_inherit_count,
            "suppressed_nested_render_count": suppressed_persistent_nested_render_count,
            "tail_extended_count": persistent_tail_extended_count,
        },
        "translation_consistency": {
            "overridden_count": translation_consistency_override_count,
            "policy": "same_corrected_source_classification_design_role_majority_then_confidence",
        },
        "vietnamese_render_deduplication": {
            "suppressed_duplicate_count": suppressed_duplicate_render_count,
        },
        "vietnamese_render_overlap_guard": {
            "passed": True,
            "conflict_count": 0,
            "minimum_box_affinity": 0.72,
        },
        "lite_parallel_triage": lite_triage,
        "strong_review_batching": {
            "batch_size": GEMINI_REVIEW_BATCH_EVENTS,
            "batch_count": (
                math.ceil(semantic_count / GEMINI_REVIEW_BATCH_EVENTS)
                if semantic_count
                else 0
            ),
            "editorial_ui_second_review_count": len(
                editorial_ui_second_review_ids
            ),
            "uncertain_second_review_count": len(
                uncertain_second_review_ids
            ),
            "focused_second_review_batch_count": (
                focused_second_review_batch_count
            ),
            "focused_ocr_pre_gate_skipped_count": len(
                focused_ocr_pre_gate_skipped_ids
            ),
            "focused_ocr_pre_gate_skipped_ids": sorted(
                focused_ocr_pre_gate_skipped_ids
            ),
            "focused_ocr_guard_rejected_count": len(
                focused_ocr_guard_rejected_ids
            ),
            "focused_ocr_guard_rejected_ids": sorted(
                focused_ocr_guard_rejected_ids
            ),
            "parallel": strong_parallel,
        },
        "local_speech_routing": {
            "matched_event_count": len(local_spoken_decisions),
            "gemini_input_event_count": gemini_input_profile["event_count"],
            "minimum_combined_score": LOCAL_SPOKEN_ROUTE_MIN_SCORE,
            "maximum_duration_ratio": LOCAL_SPOKEN_ROUTE_MAX_DURATION_RATIO,
            "rejected_temporal_shape_count": int(
                speech_review_routing.get("rejected_temporal_shape_count", 0)
            ),
            "minimum_text_length_ratio": LOCAL_SPOKEN_ROUTE_MIN_TEXT_LENGTH_RATIO,
            "rejected_text_shape_count": int(
                speech_review_routing.get("rejected_text_shape_count", 0)
            ),
            "policy": "trusted_text_time_duration_and_text_shape_match_remove_only",
        },
        "local_weak_ocr_fragment_routing": {
            "deferred_event_count": len(local_weak_ocr_fragment_decisions),
            "gemini_input_event_count": gemini_input_profile["event_count"],
            "max_duration_seconds": PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_DURATION_SECONDS,
            "max_observed_frames": PRIMARY_REVIEW_WEAK_FRAGMENT_MAX_OBSERVED_FRAMES,
            "policy": "blank_few_frame_detector_fragment_deferred_to_visual_owner",
        },
        "semantic_fragmentation_guard": fragmentation_guard,
        "local_short_small_middle_routing": {
            "enabled": normalized_config["middle_microtext_filter_enabled"],
            "kept_event_count": len(local_middle_microtext_decisions),
            "gemini_input_event_count": gemini_input_profile["event_count"],
            "policy": "keep_short_small_middle_without_gemini",
            "max_duration_seconds": normalized_config[
                "middle_microtext_max_duration_seconds"
            ],
            "max_area_fraction": normalized_config[
                "middle_microtext_max_area_fraction"
            ],
            "max_height_fraction": normalized_config[
                "middle_microtext_max_height_fraction"
            ],
            "middle_top_fraction": normalized_config[
                "middle_microtext_top_fraction"
            ],
            "middle_bottom_fraction": normalized_config[
                "middle_microtext_bottom_fraction"
            ],
        },
        "scenery_guard": {
            "model": SCENERY_GUARD_MODEL,
            "enabled": normalized_config["scenery_guard_enabled"],
            "candidate_count": len(guard_results),
            "advisory_only": True,
            "lite_not_text_count": len(lite_not_text_ids),
            "protected_false_positive_count": len(main_model_preserved_ids),
            "main_model_preserved_count": len(main_model_preserved_ids),
            "main_model_selected_count": len(main_model_selected_ids),
            "not_text_confidence_threshold": SCENERY_NOT_TEXT_CONFIDENCE,
        },
    }


def request_kaggle_proxy_review(
    *,
    profile: dict[str, Any],
    profile_path: str | Path,
    config: object,
    work_folder: str | Path,
    srt_path: str | Path | None = None,
    client: object,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Review downloaded VSR evidence through the local Kaggle Model Proxy.

    The proxy's OpenAI-compatible endpoint accepts contact-sheet images but
    does not expose Gemini Files API uploads.  Timeline, OCR, geometry and an
    optional SRT therefore travel in the prompt while representative visual
    evidence is attached as image data URLs.
    """

    normalized_config = normalize_localization_config(config)
    if not normalized_config["enabled"]:
        raise SilentTextLocalizationError("Silent-text localization is disabled.")
    logger = log or (lambda _message: None)
    srt = Path(srt_path) if srt_path else None
    srt_text = srt.read_text(encoding="utf-8-sig") if srt and srt.is_file() else ""
    events = list(profile.get("events") or [])
    if not events:
        raise SilentTextLocalizationError("Subtitle event profile has no events.")
    # Gemini 3.6 uses a substantial reasoning budget for multimodal review.
    # One contact sheet per request keeps the highest-quality model reliable
    # and still preserves eight neighboring events as local timeline context.
    batch_size = CONTACT_SHEET_EVENTS
    decisions: list[dict[str, Any]] = []
    used_models: list[str] = []
    total_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    batch_count = math.ceil(len(events) / batch_size)
    for batch_index, start in enumerate(range(0, len(events), batch_size), start=1):
        batch_profile = dict(profile)
        batch_profile["events"] = events[start : start + batch_size]
        batch_profile["event_count"] = len(batch_profile["events"])
        sheets = build_contact_sheets(
            batch_profile,
            profile_path,
            Path(work_folder)
            / "contact-sheets"
            / f"batch-{batch_index:03d}",
        )
        prompt = build_gemini_prompt(batch_profile, srt_text=srt_text)
        prompt += (
            f"\n\nĐây là lô {batch_index}/{batch_count} theo thứ tự thời gian. "
            "Nguồn bằng chứng gồm OCR, thời gian/toạ độ, các ảnh toàn khung "
            "và crop trong contact sheet"
            + (", cùng SRT nguồn." if srt_text else ".")
            + " Không được suy đoán rằng có audio/video đính kèm ngoài các "
            "bằng chứng vừa liệt kê."
        )
        checkpoint_dir = Path(work_folder) / "proxy-review-checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"batch-{batch_index:03d}.json"
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "events": batch_profile["events"],
                    "config": normalized_config,
                    "prompt": prompt,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        batch_review = None
        checkpoint = {}
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            checkpoint = {}
        if (
            checkpoint.get("fingerprint") == fingerprint
            and isinstance(checkpoint.get("response"), dict)
        ):
            try:
                batch_review = validate_gemini_decisions(
                    batch_profile,
                    checkpoint["response"],
                    normalized_config,
                )
                cached_model = str(checkpoint.get("model") or "").strip()
                if cached_model:
                    used_models.append(cached_model)
                cached_usage = checkpoint.get("usage") or {}
                for key in total_usage:
                    total_usage[key] += int(cached_usage.get(key) or 0)
                logger(
                    f"Reusing verified Kaggle Gemini batch "
                    f"{batch_index}/{batch_count}."
                )
            except SilentTextLocalizationError:
                batch_review = None
        for attempt in range(2):
            if batch_review is not None:
                break
            response = client.generate(
                prompt=(
                    prompt
                    if attempt == 0
                    else (
                        prompt
                        + "\n\nLần trả lời trước không vượt qua kiểm tra an "
                        "toàn. Hãy trả lại toàn bộ item đúng track_id, chỉ JSON "
                        "và tuân thủ mọi quy tắc."
                    )
                ),
                images=sheets,
            )
            response = _normalize_proxy_response(
                response,
                batch_profile,
            )
            try:
                batch_review = validate_gemini_decisions(
                    batch_profile,
                    response,
                    normalized_config,
                )
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "model": str(
                                getattr(client, "last_model", "") or ""
                            ),
                            "usage": dict(
                                getattr(client, "last_usage", {}) or {}
                            ),
                            "response": response,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                break
            except SilentTextLocalizationError:
                if attempt:
                    raise
                logger(
                    f"Kaggle Gemini batch {batch_index}/{batch_count} failed "
                    "contract validation; retrying once."
                )
        if batch_review is None:
            raise SilentTextLocalizationError(
                "Kaggle Gemini response did not pass contract validation."
            )
        decisions.extend(batch_review["decisions"])
        if checkpoint.get("fingerprint") != fingerprint or not checkpoint.get(
            "response"
        ):
            proxy_model = str(
                getattr(client, "last_model", "") or ""
            ).strip()
            if proxy_model:
                used_models.append(proxy_model)
            usage = dict(getattr(client, "last_usage", {}) or {})
            for key in total_usage:
                total_usage[key] += int(usage.get(key) or 0)
            logger(
                f"Kaggle Gemini batch {batch_index}/{batch_count} completed "
                f"with {proxy_model or 'unknown model'}."
            )

    _link_short_text_fragments(profile, decisions)
    decisions.sort(
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
            item["track_id"],
        )
    )
    review = {
        "schema": REVIEW_SCHEMA,
        "version": REVIEW_VERSION,
        "source_video_name": profile.get("source_video_name") or "",
        "config": normalized_config,
        "summary": _review_summary(decisions),
        "decisions": decisions,
        "ai_provider": "kaggle-model-proxy",
        "ai_model": (
            used_models[0]
            if used_models and len(set(used_models)) == 1
            else "mixed"
        ),
        "ai_models_by_batch": used_models,
        "ai_usage": total_usage,
        "evidence_scope": _gemini_evidence_scope(srt_text=srt_text),
    }
    logger("Kaggle Gemini review completed successfully.")
    return review


def _normalize_proxy_response(
    response: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Normalize harmless OpenAI-proxy aliases before strict validation."""

    items = response.get("items")
    if not isinstance(items, list):
        return response
    event_by_id = {
        str(event.get("track_id") or ""): event
        for event in profile.get("events") or []
    }
    normalized_items = []
    for raw in items:
        if not isinstance(raw, dict):
            normalized_items.append(raw)
            continue
        item = dict(raw)
        if "classification" not in item and "label" in item:
            item["classification"] = item.get("label")
        if "text_origin" not in item and "origin" in item:
            item["text_origin"] = item.get("origin")
        if "replacement_action" not in item and "action" in item:
            item["replacement_action"] = item.get("action")
        if not isinstance(item.get("vietnamese_translation"), str):
            item["vietnamese_translation"] = ""
        if not isinstance(item.get("highlight_terms"), list):
            item["highlight_terms"] = []
        classification = str(item.get("classification") or "")
        if not item.get("replacement_action"):
            if classification == "spoken_subtitle":
                item["replacement_action"] = "keep_removed"
            elif (
                classification in TRANSLATABLE_CLASSES
                and str(item.get("vietnamese_translation") or "").strip()
            ):
                item["replacement_action"] = "replace_with_vietnamese"
            elif classification == "uncertain":
                item["replacement_action"] = "needs_review"
            elif classification == "decorative_overlay":
                item["replacement_action"] = "ignore"
        track_id = str(item.get("track_id") or "")
        event = event_by_id.get(track_id) or {}
        if not isinstance(item.get("source_text_corrected"), str):
            item["source_text_corrected"] = str(
                event.get("ocr_best_text") or ""
            )
        normalized_items.append(item)
    return {"items": normalized_items}
