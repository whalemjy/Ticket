import json
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocess import preprocess_input
from utils.recognition_validation import (
    filter_continuous_recoveries,
    is_non_entry_status_text,
    normalized_text,
    sequence_value,
    split_merged_sequence_entry,
)

PROJECT_ROOT = Path(__file__).resolve().parent
# TODO: Replace this directory after the storage layout is decided.
MODEL_ROOT = PROJECT_ROOT / "models"
DET_MODEL_DIR = MODEL_ROOT / "PP-OCRv6_medium_det_infer"
REC_MODEL_DIR = MODEL_ROOT / "PP-OCRv6_medium_rec_infer"
INTER_DIR = PROJECT_ROOT / "inter"
TEXT_CROP_DIR = INTER_DIR / "text_crops"
OCR_RESULT_DIR = INTER_DIR / "ocr_results"
COMMAND_DIR = Path("./command")
INPUT_IMAGE = Path("./assets/pdfs/110kV天香变电站.pdf")
PENDING_IMAGES = []

# The sequence numbers are in the same narrow column as the "顺序" header.
SEQUENCE_X_TOLERANCE_RATIO = 0.06
# Ignore the check-mark column on the right side of the operation table.
COMMAND_RIGHT_BOUNDARY_RATIO = 0.89
REC_BATCH_SIZE = 4
ENHANCE_TEXT_CROPS = True
TEXT_CLAHE_CLIP_LIMIT = 1.8
TEXT_SHARPEN_AMOUNT = 0.3
OPERATION_TASK_MIN_SIMILARITY = 0.8
OPERATION_TASK_MIN_CONFIDENCE = 0.75
LOCAL_SEQUENCE_MIN_CONFIDENCE = 0.5
# A geometry fallback may fill only small internal gaps.  It never invents a
# page's first sequence number and never repairs duplicate or ambiguous rows.
SEQUENCE_GEOMETRY_FALLBACK_MAX_GAP = 2
UPLOAD_ORDER_ERROR = "请按照正确顺序上传操作票或拍摄更清晰的操作票"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class TicketRecognitionError(ValueError):
    """A user-facing parsing failure with an optional source-page location."""

    def __init__(self, region, detail, *, page_path=None):
        self.region = region
        self.detail = detail
        self.page_path = page_path
        super().__init__(detail)

    def __str__(self):
        page = f"页面：{Path(self.page_path).name}\n" if self.page_path else ""
        return (
            f"操作票识别失败\n{page}问题区域：{self.region}\n"
            f"具体原因：{self.detail}\n{UPLOAD_ORDER_ERROR}"
        )


def _anchor_location(anchor):
    return (
        f"识别框位置 x={anchor['left']:.0f}～{anchor['right']:.0f}，"
        f"y={anchor['top']:.0f}～{anchor['bottom']:.0f}（像素）"
    )


def crop_text_region(image, points):
    """Crop and rectify one quadrilateral text region."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected four (x, y) points, got shape {points.shape}")

    crop_width = max(
        int(round(np.linalg.norm(points[0] - points[1]))),
        int(round(np.linalg.norm(points[2] - points[3]))),
        1,
    )
    crop_height = max(
        int(round(np.linalg.norm(points[0] - points[3]))),
        int(round(np.linalg.norm(points[1] - points[2]))),
        1,
    )
    target_points = np.array(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, target_points)
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return crop


def _write_png(image_path, image):
    """Write a PNG through NumPy so Windows Unicode paths remain intact."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed to encode PNG image: {image_path}")
    encoded.tofile(image_path)


def _enhance_text_crop(crop):
    """Enhance a small OCR crop without changing its dimensions or color mode."""
    lightness, channel_a, channel_b = cv2.split(
        cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    )
    lightness = cv2.createCLAHE(
        clipLimit=TEXT_CLAHE_CLIP_LIMIT,
        tileGridSize=(2, 2),
    ).apply(lightness)
    contrast_enhanced = cv2.cvtColor(
        cv2.merge((lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=0.8)
    return cv2.addWeighted(
        contrast_enhanced,
        1.0 + TEXT_SHARPEN_AMOUNT,
        blurred,
        -TEXT_SHARPEN_AMOUNT,
        0,
    )


def save_text_regions(result, output_dir, source_path):
    """Save OCR-ready crops for every detected region."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = result["input_img"].copy()
    source_name = Path(source_path).stem
    saved_paths = []

    for region_index, points in enumerate(result["dt_polys"]):
        crop = crop_text_region(source_image, points)
        if ENHANCE_TEXT_CROPS:
            crop = _enhance_text_crop(crop)
        crop_path = output_dir / f"{source_name}_{region_index:03d}.png"
        _write_png(crop_path, crop)
        saved_paths.append(crop_path)

    return saved_paths


def _region_geometry(points):
    points = np.asarray(points, dtype=np.float32)
    left = float(points[:, 0].min())
    right = float(points[:, 0].max())
    top = float(points[:, 1].min())
    bottom = float(points[:, 1].max())
    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "center_x": (left + right) / 2,
        "center_y": (top + bottom) / 2,
    }


def recognize_text_regions(rec_model, crop_paths, polygons, detection_scores):
    """Recognize crops in batches while preserving detection-box associations."""
    crop_paths = list(crop_paths)
    polygons = list(polygons)
    detection_scores = list(detection_scores)
    if len(crop_paths) != len(polygons):
        raise ValueError(
            "Crop path and polygon counts differ: "
            f"{len(crop_paths)} != {len(polygons)}"
        )
    if len(detection_scores) != len(polygons):
        raise ValueError(
            "Detection score and polygon counts differ: "
            f"{len(detection_scores)} != {len(polygons)}"
        )
    if not crop_paths:
        return []

    predictions = rec_model.predict(
        [str(crop_path) for crop_path in crop_paths],
        batch_size=REC_BATCH_SIZE,
    )
    if len(predictions) != len(polygons):
        raise ValueError(
            "Recognition result and polygon counts differ: "
            f"{len(predictions)} != {len(polygons)}"
        )

    records = []
    for region_index, (prediction, points, detection_score) in enumerate(
        zip(predictions, polygons, detection_scores)
    ):
        text = str(prediction["rec_text"]).strip()
        try:
            score = float(prediction["rec_score"])
        except (KeyError, TypeError):
            score = None

        record = {
            "index": region_index,
            "text": text,
            "score": score,
            "det_score": float(detection_score),
            "box": np.asarray(points, dtype=int).tolist(),
        }
        record.update(_region_geometry(points))
        records.append(record)
    return records


def _annotation_font(size):
    font_paths = (
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    )
    for font_path in font_paths:
        if Path(font_path).is_file():
            return ImageFont.truetype(font_path, size)
    raise RuntimeError("OCR annotation requires a Chinese-capable font")


def _wrap_annotation_text(draw, text, font, max_width):
    lines = []
    line = ""
    for character in text or "(empty)":
        if character == "\n":
            lines.append(line)
            line = ""
        elif line and draw.textlength(line + character, font=font) > max_width:
            lines.append(line)
            line = character
        else:
            line += character
    lines.append(line)
    return lines


def save_recognition_results(page_path, image, records, output_dir=OCR_RESULT_DIR):
    """Save OCR data and a numbered image before ticket parsing can fail."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(page_path).stem
    json_path = output_dir / f"{stem}.json"
    image_path = output_dir / f"{stem}_annotated.png"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {"source_image": str(page_path), "regions": records},
            output_file,
            ensure_ascii=False,
            indent=2,
        )
        output_file.write("\n")

    source = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    sidebar_width = max(800, min(1400, source.width // 2))
    font_size = max(20, min(30, source.width // 100))
    font = _annotation_font(font_size)
    line_height = font_size + 8
    padding = 24
    text_width = sidebar_width - 2 * padding
    measure = ImageDraw.Draw(source)
    entries = []
    for record in records:
        score = "n/a" if record["score"] is None else f'{record["score"]:.3f}'
        title = (
            f'#{record["index"]:03d}  OCR {score}  '
            f'DET {record["det_score"]:.3f}'
        )
        lines = _wrap_annotation_text(measure, record["text"], font, text_width)
        entries.append((title, lines))

    entry_heights = [(len(lines) + 1) * line_height + 16 for _, lines in entries]
    canvas_height = max(source.height, 2 * padding + sum(entry_heights))
    canvas = Image.new("RGB", (source.width + sidebar_width, canvas_height), "white")
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.line(
        (source.width, 0, source.width, canvas_height),
        fill="#b8b8b8",
        width=2,
    )

    for record in records:
        points = [tuple(point) for point in record["box"]]
        draw.line(points + [points[0]], fill="#d3382b", width=3)
        x = min(max(int(record["right"]) + 4, 0), source.width - 1)
        y = min(max(int(record["top"]), 0), source.height - font_size)
        draw.text(
            (x, y),
            f'{record["index"]:03d}',
            font=font,
            fill="#b52118",
            stroke_width=2,
            stroke_fill="white",
        )

    y = padding
    for (title, lines), height in zip(entries, entry_heights):
        x = source.width + padding
        draw.text((x, y), title, font=font, fill="#b52118")
        for line_index, line in enumerate(lines, start=1):
            draw.text(
                (x, y + line_index * line_height),
                line,
                font=font,
                fill="#202020",
            )
        y += height
        draw.line((x, y - 8, canvas.width - padding, y - 8), fill="#dedede")

    canvas.save(image_path, format="PNG")
    return json_path, image_path


def _normalized_text(text):
    return normalized_text(text)


def find_text_anchor(
    records,
    target_text,
    min_similarity=1.0,
    *,
    min_confidence=None,
    prefer_confidence=False,
):
    """Find the best OCR box for a fixed label, optionally allowing OCR errors."""
    normalized_target = _normalized_text(target_text)
    candidates = []
    for record in records:
        score = record["score"]
        if min_confidence is not None and (
            score is None or score < min_confidence
        ):
            continue
        normalized = _normalized_text(record["text"])
        if normalized_target in normalized:
            similarity = 1.0
        else:
            similarity = SequenceMatcher(None, normalized_target, normalized).ratio()
        if similarity >= min_similarity:
            candidates.append((similarity, record))

    if not candidates:
        confidence_note = (
            f"（要求识别置信度至少为 {min_confidence:.0%}）"
            if min_confidence is not None else ""
        )
        raise TicketRecognitionError(
            f"“{target_text}”标签",
            f"未找到可用的“{target_text}”文字标签{confidence_note}，"
            "请检查该区域是否被裁切、遮挡或识别错误。",
        )
    if prefer_confidence:
        return max(
            candidates,
            key=lambda item: (item[1]["score"] or 0.0, item[0]),
        )[1]
    return max(
        candidates,
        key=lambda item: (item[0], item[1]["score"] or 0.0),
    )[1]


def find_unit_anchor(records):
    """Locate a misrecognized unit label using the surrounding header layout."""
    try:
        return find_text_anchor(records, "单位")
    except ValueError as original_error:
        try:
            issuer = find_text_anchor(records, "发令人")
            start_time = find_text_anchor(records, "操作开始时间")
            number = find_text_anchor(records, "编号")
        except ValueError:
            raise original_error from None

        if not issuer["center_y"] < start_time["center_y"]:
            raise original_error

        number_height = max(number["bottom"] - number["top"], 1.0)
        same_row_to_left = sorted(
            (
                record
                for record in records
                if record["right"] < number["left"]
                and abs(record["center_y"] - number["center_y"])
                <= 0.75
                * max(number_height, record["bottom"] - record["top"], 1.0)
            ),
            key=lambda record: record["center_x"],
        )
        if len(same_row_to_left) < 2:
            raise original_error

        unit = same_row_to_left[-2]
        if not unit["center_y"] < issuer["center_y"]:
            raise original_error
        for lower_label in (issuer, start_time):
            label_width = max(lower_label["right"] - lower_label["left"], 1.0)
            unit_width = max(unit["right"] - unit["left"], 1.0)
            if abs(unit["center_x"] - lower_label["center_x"]) > (
                0.65 * max(label_width, unit_width)
            ):
                raise original_error
        return unit


def find_operation_task_anchor(records):
    """Find the mission label, falling back to the nearest box above '顺序'."""
    sequence_anchor = find_text_anchor(records, "顺序")
    header_records = [
        record
        for record in records
        if record["bottom"] <= sequence_anchor["top"]
    ]
    try:
        return find_text_anchor(
            header_records,
            "操作任务",
            min_similarity=OPERATION_TASK_MIN_SIMILARITY,
            min_confidence=OPERATION_TASK_MIN_CONFIDENCE,
            prefer_confidence=True,
        )
    except ValueError as original_error:
        candidates = [
            record
            for record in header_records
            if _normalized_text(record["text"])
        ]
        if not candidates:
            raise original_error

        return min(
            candidates,
            key=lambda record: (
                sequence_anchor["top"] - record["bottom"],
                -record["det_score"],
                abs(record["center_x"] - sequence_anchor["center_x"]),
            ),
        )


def _mission_label_prefix(text):
    """Return a mission-label prefix only when it starts the OCR text."""
    normalized = _normalized_text(text)
    for prefix in ("操作任务", "操作务", "操作", "任务"):
        if normalized.startswith(prefix):
            return prefix
    return None


def _vertical_gap(record, band_top, band_bottom):
    if record["bottom"] < band_top:
        return band_top - record["bottom"]
    if record["top"] > band_bottom:
        return record["top"] - band_bottom
    return 0.0


def _find_mission_label_records(records, sequence_anchor):
    """Find the label boxes immediately above the operation table header."""
    sequence_width = sequence_anchor["right"] - sequence_anchor["left"]
    label_left_limit = sequence_anchor["right"] + max(sequence_width * 0.25, 5.0)
    candidates = [
        record
        for record in records
        if record["center_y"] < sequence_anchor["top"]
        and record["left"] <= label_left_limit
        and _mission_label_prefix(record["text"]) is not None
    ]
    if not candidates:
        return []

    # "操作开始时间" is also in the left column. Start with the closest
    # label-like box above "顺序", then only join vertically adjacent parts.
    seed = max(candidates, key=lambda record: record["bottom"])
    selected = [seed]
    selected_ids = {seed["index"]}
    band_top = seed["top"]
    band_bottom = seed["bottom"]

    changed = True
    while changed:
        changed = False
        band_height = max(band_bottom - band_top, 1.0)
        for record in candidates:
            if record["index"] in selected_ids:
                continue
            record_height = max(record["bottom"] - record["top"], 1.0)
            allowed_gap = max(min(band_height, record_height) * 0.3, 3.0)
            if _vertical_gap(record, band_top, band_bottom) > allowed_gap:
                continue
            selected.append(record)
            selected_ids.add(record["index"])
            band_top = min(band_top, record["top"])
            band_bottom = max(band_bottom, record["bottom"])
            changed = True

    return selected


def _find_check_column_left(records, sequence_anchor):
    """Locate the check column from the operation-table header row."""
    sequence_width = max(sequence_anchor["right"] - sequence_anchor["left"], 1.0)
    row_records = [
        record
        for record in records
        if record["top"] <= sequence_anchor["bottom"]
        and record["bottom"] >= sequence_anchor["top"]
        and record["center_x"] > sequence_anchor["right"]
    ]
    check_texts = {"√", "✓", "✔", "∨"}
    recognized_checks = [
        record
        for record in row_records
        if _normalized_text(record["text"]) in check_texts
    ]
    if recognized_checks:
        return min(record["left"] for record in recognized_checks)

    narrow_records = [
        record
        for record in row_records
        if record["right"] - record["left"] <= sequence_width * 1.5
    ]
    if narrow_records:
        return max(narrow_records, key=lambda record: record["center_x"])["left"]
    return float("inf")


def _is_operation_mode_text(text):
    normalized = _normalized_text(text)
    return re.fullmatch(
        r"[()（）√✓✔∨]*(?:监护下操作|单人操作|检修人员操作)",
        normalized,
    ) is not None


def extract_mission(records):
    """Extract a one- or multi-line mission without a fixed right boundary."""
    sequence_anchor = find_text_anchor(records, "顺序")
    header_records = [
        record
        for record in records
        if record["center_y"] < sequence_anchor["top"]
    ]
    check_column_left = _find_check_column_left(records, sequence_anchor)
    mission_header_records = [
        record
        for record in header_records
        if record["left"] < check_column_left
        and not _is_operation_mode_text(record["text"])
    ]
    label_records = _find_mission_label_records(header_records, sequence_anchor)

    # Keep the established layout fallback for labels whose OCR text is too
    # damaged to expose either "操作" or "任务".
    if not label_records:
        label_anchor = find_operation_task_anchor(records)
        return extract_field_value(
            mission_header_records,
            "操作任务",
            label_anchor=label_anchor,
        )

    prefixes = {
        _mission_label_prefix(record["text"])
        for record in label_records
    }
    has_complete_label = (
        "操作任务" in prefixes
        or "操作务" in prefixes
        or ("操作" in prefixes and "任务" in prefixes)
    )
    if not has_complete_label:
        label_anchor = find_operation_task_anchor(records)
        return extract_field_value(
            mission_header_records,
            "操作任务",
            label_anchor=label_anchor,
        )

    band_top = min(record["top"] for record in label_records)
    band_bottom = max(record["bottom"] for record in label_records)
    label_heights = [
        max(record["bottom"] - record["top"], 1.0)
        for record in label_records
    ]
    band_padding = max(float(np.median(label_heights)) * 0.25, 3.0)
    label_column_right = sequence_anchor["right"]

    mission_records = [
        record
        for record in mission_header_records
        if record["top"] <= band_bottom + band_padding
        and record["bottom"] >= band_top - band_padding
        and _normalized_text(record["text"])
    ]
    mission_records.sort(key=lambda item: (item["center_y"], item["left"]))

    parts = []
    for record in mission_records:
        text = _normalized_text(record["text"])
        prefix = None
        if record["left"] <= label_column_right:
            prefix = _mission_label_prefix(text)
        if prefix is not None:
            text = text[len(prefix) :]
        elif record["center_x"] <= label_column_right:
            continue

        # A detector box may extend into the dynamically located check column.
        # Only remove a recognized check suffix; never reject the long text box.
        if record["right"] >= check_column_left:
            text = re.sub(r"[√✓✔∨]+$", "", text)
        if text:
            parts.append(text)

    mission = "".join(parts)
    if not mission:
        raise TicketRecognitionError(
            "操作任务内容",
            "已找到操作任务标签，但未识别到右侧的任务内容。",
        )
    return mission


def extract_field_value(
    records,
    label_text,
    *,
    right_label_text=None,
    right_boundary=None,
    min_label_similarity=1.0,
    label_anchor=None,
):
    """Extract text to the right of a label within the same table row."""
    label = label_anchor or find_text_anchor(
        records,
        label_text,
        min_label_similarity,
    )
    if right_boundary is None:
        if right_label_text is None:
            right_boundary = float("inf")
        else:
            right_boundary = find_text_anchor(records, right_label_text)["left"]

    row_padding = max((label["bottom"] - label["top"]) * 0.15, 1.0)
    value_records = [
        record
        for record in records
        if record["center_x"] > label["right"]
        and record["center_x"] < right_boundary
        and label["top"] - row_padding
        <= record["center_y"]
        <= label["bottom"] + row_padding
        and _normalized_text(record["text"])
    ]
    value_records.sort(key=lambda item: (item["center_y"], item["left"]))
    value = "".join(_normalized_text(record["text"]) for record in value_records)
    if not value:
        raise TicketRecognitionError(
            f"“{label_text}”右侧内容",
            f"已找到“{label_text}”标签，但同一行右侧未识别到有效内容；"
            f"{_anchor_location(label)}。",
        )
    return value


def _sequence_value(text, *, allow_wrappers=False):
    """Compatibility wrapper around :func:`utils.recognition_validation.sequence_value`."""
    return sequence_value(text, allow_wrappers=allow_wrappers)
    """Legacy local wrapper retained for callers inside this module."""
    '''
    normalized = _normalized_text(text)
    if re.fullmatch(r"\\d{1,3}", normalized):
        return int(normalized)
    if allow_wrappers:
        match = re.fullmatch(
            r"[.:：。·,，;；|丨!！]*([0-9]{1,3})[.:：。·,，;；|丨!！]*",
            normalized,
        )
        if match is not None:
            return int(match.group(1))
    return None
    '''


def _is_non_entry_status_text(text):
    """Compatibility wrapper around the shared status-text validator."""
    return is_non_entry_status_text(text)
    '''
    normalized = _normalized_text(text).strip(
        "()（）[]【】.:：。·,，;；|丨!！√✓✔∨V"
    )
    return normalized in NON_ENTRY_STATUS_TEXTS
    '''


def _split_merged_sequence_entry(record, sequence_anchor):
    """Compatibility wrapper around the shared merged-entry validator."""
    return split_merged_sequence_entry(record, sequence_anchor)
    '''
    """Split a sequence prefix only when its box crosses the sequence column."""
    normalized = _normalized_text(record["text"])
    match = re.fullmatch(r"(\\d{1,3})(\\D.*)", normalized)
    if match is None:
        return None

    crosses_sequence_column = (
        record["left"]
        <= sequence_anchor["center_x"]
        <= record["right"]
    )
    extends_into_content_column = record["right"] > sequence_anchor["right"]
    if not crosses_sequence_column or not extends_into_content_column:
        return None

    return int(match.group(1)), match.group(2)
    '''


def _group_operation_content_rows(records, sequence_anchor, image_shape, sequence_records):
    """Group operation-text boxes into rows without relying on sequence boxes."""
    image_height, image_width = image_shape[:2]
    command_right = image_width * COMMAND_RIGHT_BOUNDARY_RATIO
    sequence_record_ids = {
        record["index"] for _, record, _ in sequence_records
    }
    sequence_centers = sorted(
        record["center_y"] for _, record, _ in sequence_records
    )
    if sequence_centers:
        max_center_y = sequence_centers[-1] + image_height * 0.08
    else:
        max_center_y = image_height

    check_texts = {"V", "√", "✓", "✔", "∨"}
    content_records = [
        record
        for record in records
        if record["index"] not in sequence_record_ids
        and sequence_anchor["bottom"] < record["center_y"] <= max_center_y
        and record["center_x"] > sequence_anchor["right"]
        and record["center_x"] < command_right
        and _normalized_text(record["text"])
        and _normalized_text(record["text"]) not in check_texts
        and not _is_non_entry_status_text(record["text"])
        and _sequence_value(record["text"], allow_wrappers=True) is None
    ]
    content_records.sort(key=lambda record: (record["center_y"], record["left"]))
    if not content_records:
        return []

    heights = [
        max(record["bottom"] - record["top"], 1.0)
        for record in content_records
    ]
    cluster_tolerance = max(float(np.median(heights)) * 0.5, image_height * 0.003)
    rows = []
    for record in content_records:
        if (
            not rows
            or record["center_y"] - rows[-1]["center_y"] > cluster_tolerance
        ):
            rows.append(
                {
                    "records": [record],
                    "top": record["top"],
                    "bottom": record["bottom"],
                    "center_y": record["center_y"],
                }
            )
            continue

        row = rows[-1]
        row["records"].append(record)
        row["top"] = min(row["top"], record["top"])
        row["bottom"] = max(row["bottom"], record["bottom"])
        row["center_y"] = float(
            np.median([item["center_y"] for item in row["records"]])
        )
    return rows


def _crop_overlaps_sequence_record(x1, y1, x2, y2, sequence_record):
    return (
        x1 < sequence_record["right"]
        and x2 > sequence_record["left"]
        and y1 < sequence_record["bottom"]
        and y2 > sequence_record["top"]
    )


def _filter_continuous_recoveries(sequence_records, recovered):
    """Keep recovered values only when each primary-sequence gap is complete."""
    recovered = sorted(recovered, key=lambda item: item[1]["center_y"])
    if not recovered:
        return []

    existing = sorted(sequence_records, key=lambda item: item[1]["center_y"])
    if not existing:
        values = [sequence for sequence, _, _ in recovered]
        if values[0] < 1 or any(
            current != previous + 1
            for previous, current in zip(values, values[1:])
        ):
            return []
        return recovered

    accepted = []

    first_sequence, first_record, _ = existing[0]
    before = [
        item
        for item in recovered
        if item[1]["center_y"] < first_record["center_y"]
    ]
    before_values = [sequence for sequence, _, _ in before]
    expected_before = list(
        range(first_sequence - len(before_values), first_sequence)
    )
    if expected_before and expected_before[0] >= 1 and before_values == expected_before:
        accepted.extend(before)

    for previous, current in zip(existing, existing[1:]):
        previous_sequence, previous_record, _ = previous
        current_sequence, current_record, _ = current
        between = [
            item
            for item in recovered
            if previous_record["center_y"]
            < item[1]["center_y"]
            < current_record["center_y"]
        ]
        between_values = [sequence for sequence, _, _ in between]
        expected_between = list(
            range(previous_sequence + 1, current_sequence)
        )
        if between_values == expected_between:
            accepted.extend(between)

    last_sequence, last_record, _ = existing[-1]
    after = [
        item
        for item in recovered
        if item[1]["center_y"] > last_record["center_y"]
    ]
    after_values = [sequence for sequence, _, _ in after]
    expected_after = list(
        range(last_sequence + 1, last_sequence + 1 + len(after_values))
    )
    if after_values == expected_after:
        accepted.extend(after)

    return accepted


def _recover_missing_sequence_records(
    records,
    sequence_anchor,
    sequence_records,
    image_shape,
    source_image,
    rec_model,
):
    """Recognize sequence cells for content rows whose number box was missed."""
    content_rows = _group_operation_content_rows(
        records,
        sequence_anchor,
        image_shape,
        sequence_records,
    )
    if not content_rows:
        return []

    row_centers = [row["center_y"] for row in content_rows]
    row_differences = [
        row_centers[index] - row_centers[index - 1]
        for index in range(1, len(row_centers))
        if row_centers[index] > row_centers[index - 1]
    ]
    typical_spacing = (
        float(np.median(row_differences))
        if row_differences
        else image_shape[0] * 0.03
    )
    row_heights = [max(row["bottom"] - row["top"], 1.0) for row in content_rows]
    alignment_tolerance = max(
        typical_spacing * 0.35,
        float(np.median(row_heights)) * 0.75,
        image_shape[0] * 0.006,
    )
    existing_centers = [
        record["center_y"] for _, record, _ in sequence_records
    ]
    missing_rows = [
        row
        for row in content_rows
        if not any(
            abs(row["center_y"] - center_y) <= alignment_tolerance
            for center_y in existing_centers
        )
    ]
    if not missing_rows:
        return []

    image_height, image_width = image_shape[:2]
    sequence_width = max(sequence_anchor["right"] - sequence_anchor["left"], 1.0)
    x1 = max(int(sequence_anchor["left"] - sequence_width * 0.1), 0)
    x2 = min(int(sequence_anchor["right"] + sequence_width * 0.1), image_width)
    cell_crops = []
    crop_rows = []
    crop_geometries = []
    for row in missing_rows:
        row_height = max(row["bottom"] - row["top"], 1.0)
        y1 = max(int(row["top"] - row_height * 0.25), 0)
        y2 = min(int(row["bottom"] + row_height * 0.25), image_height)
        if any(
            _crop_overlaps_sequence_record(x1, y1, x2, y2, record)
            for _, record, _ in sequence_records
        ):
            continue
        cell_crops.append(source_image[y1:y2, x1:x2])
        crop_rows.append(row)
        crop_geometries.append((y1, y2, row["center_y"]))

    if not cell_crops:
        return []

    predictions = rec_model.predict(cell_crops, batch_size=REC_BATCH_SIZE)
    if len(predictions) != len(crop_rows):
        raise ValueError(
            "Local sequence recognition result and row counts differ: "
            f"{len(predictions)} != {len(crop_rows)}"
        )

    recovered = []
    for recovery_index, (prediction, geometry) in enumerate(
        zip(predictions, crop_geometries)
    ):
        try:
            score = float(prediction["rec_score"])
        except (KeyError, TypeError):
            score = 0.0
        sequence = _sequence_value(
            prediction["rec_text"],
            allow_wrappers=True,
        )
        if sequence is None or score < LOCAL_SEQUENCE_MIN_CONFIDENCE:
            continue

        y1, y2, center_y = geometry
        synthetic_record = {
            "index": -(recovery_index + 1),
            "text": str(sequence),
            "score": score,
            "det_score": 0.0,
            "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "left": float(x1),
            "right": float(x2),
            "top": float(y1),
            "bottom": float(y2),
            "center_x": (x1 + x2) / 2,
            "center_y": center_y,
        }
        recovered.append((sequence, synthetic_record, ""))
    return filter_continuous_recoveries(sequence_records, recovered)


def _infer_missing_sequence_records(records, sequence_anchor, sequence_records, image_shape):
    """Infer small internal sequence gaps from aligned operation-text rows.

    This is deliberately conservative: every missing number must have exactly
    one content row between two recognized sequence rows.  The fallback is
    useful when the number cell is blurred or missed by detection, while a
    genuinely reordered upload still fails the continuity check later.
    """
    existing = sorted(sequence_records, key=lambda item: item[1]["center_y"])
    if len(existing) < 2:
        return []

    content_rows = _group_operation_content_rows(
        records,
        sequence_anchor,
        image_shape,
        sequence_records,
    )
    if not content_rows:
        return []

    inferred = []
    used_row_ids = set()
    existing_centers = [record[1]["center_y"] for record in existing]
    image_height = image_shape[0]
    alignment_tolerance = max(image_height * 0.012, 3.0)
    for previous, current in zip(existing, existing[1:]):
        previous_sequence, previous_record, _ = previous
        current_sequence, current_record, _ = current
        gap = current_sequence - previous_sequence - 1
        if gap <= 0 or gap > SEQUENCE_GEOMETRY_FALLBACK_MAX_GAP:
            continue

        between = [
            row
            for row in content_rows
            if previous_record["center_y"] < row["center_y"] < current_record["center_y"]
            and not any(
                abs(row["center_y"] - center_y) <= alignment_tolerance
                for center_y in existing_centers
            )
            and id(row) not in used_row_ids
        ]
        if len(between) != gap:
            continue

        interval = (
            current_record["center_y"] - previous_record["center_y"]
        ) / (gap + 1)
        position_tolerance = max(interval * 0.35, image_height * 0.012)
        if any(
            abs(
                row["center_y"]
                - (previous_record["center_y"] + interval * offset)
            )
            > position_tolerance
            for offset, row in enumerate(between, start=1)
        ):
            continue

        # A row that contains only a stray mark is not enough evidence to
        # invent a sequence number.
        if any(
            not any(
                record["center_x"] > sequence_anchor["right"]
                and _normalized_text(record["text"]).strip("V√✓✔∨")
                for record in row["records"]
            )
            for row in between
        ):
            continue

        for offset, row in enumerate(between, start=1):
            sequence = previous_sequence + offset
            row_center = row["center_y"]
            synthetic_record = {
                "index": -(len(inferred) + 1001),
                "text": str(sequence),
                "score": None,
                "det_score": 0.0,
                "box": [
                    [sequence_anchor["left"], row["top"]],
                    [sequence_anchor["right"], row["top"]],
                    [sequence_anchor["right"], row["bottom"]],
                    [sequence_anchor["left"], row["bottom"]],
                ],
                "left": sequence_anchor["left"],
                "right": sequence_anchor["right"],
                "top": row["top"],
                "bottom": row["bottom"],
                "center_x": sequence_anchor["center_x"],
                "center_y": row_center,
            }
            inferred.append((sequence, synthetic_record, ""))
            used_row_ids.add(id(row))

    return inferred


def extract_entries(records, image_shape, source_image=None, rec_model=None):
    """Extract operation item text by anchoring rows on their sequence numbers."""
    image_height, image_width = image_shape[:2]
    sequence_anchor = find_text_anchor(records, "顺序")
    x_tolerance = image_width * SEQUENCE_X_TOLERANCE_RATIO

    sequence_records = []
    for record in records:
        is_below_anchor = record["center_y"] > sequence_anchor["bottom"]
        is_in_sequence_column = (
            abs(record["center_x"] - sequence_anchor["center_x"]) <= x_tolerance
        )
        sequence = _sequence_value(
            record["text"],
            allow_wrappers=is_in_sequence_column,
        )
        if sequence is not None and is_below_anchor and is_in_sequence_column:
            sequence_records.append((sequence, record, ""))
            continue

        if not is_below_anchor:
            continue
        merged_entry = _split_merged_sequence_entry(record, sequence_anchor)
        if merged_entry is None:
            continue

        sequence, inline_text = merged_entry
        synthetic_sequence_record = dict(record)
        synthetic_sequence_record.update(
            {
                "left": sequence_anchor["left"],
                "right": sequence_anchor["right"],
                "center_x": sequence_anchor["center_x"],
            }
        )
        sequence_records.append(
            (sequence, synthetic_sequence_record, inline_text)
        )

    if source_image is not None and rec_model is not None:
        sequence_records.extend(
            _recover_missing_sequence_records(
                records,
                sequence_anchor,
                sequence_records,
                image_shape,
                source_image,
                rec_model,
            )
        )

    # Model-based recovery can still miss a blurred sequence cell.  Use the
    # row geometry only for small, one-to-one internal gaps as a final fallback.
    sequence_records.extend(
        _infer_missing_sequence_records(
            records,
            sequence_anchor,
            sequence_records,
            image_shape,
        )
    )

    sequence_records.sort(key=lambda item: item[1]["center_y"])
    if not sequence_records:
        raise TicketRecognitionError(
            "顺序列（表格左侧）",
            "已找到“顺序”表头，但表头下方未识别到任何有效序号；"
            f"表头{_anchor_location(sequence_anchor)}。",
        )

    centers = [record["center_y"] for _, record, _ in sequence_records]
    if len(centers) > 1:
        typical_row_height = float(np.median(np.diff(centers)))
    else:
        typical_row_height = image_height * 0.03

    row_boundaries = [sequence_anchor["bottom"]]
    row_boundaries.extend(
        (centers[index - 1] + centers[index]) / 2
        for index in range(1, len(centers))
    )
    row_boundaries.append(centers[-1] + typical_row_height / 2)

    command_right = image_width * COMMAND_RIGHT_BOUNDARY_RATIO
    sequence_record_ids = {
        record["index"] for _, record, _ in sequence_records
    }
    entries = {}

    for row_index, (sequence, sequence_record, inline_text) in enumerate(
        sequence_records
    ):
        row_top = row_boundaries[row_index]
        row_bottom = row_boundaries[row_index + 1]
        text_records = [
            record
            for record in records
            if record["index"] not in sequence_record_ids
            and row_top <= record["center_y"] < row_bottom
            # Detection boxes can overlap slightly across the sequence/content
            # cell border, so compare their centers instead of their edges.
            and record["center_x"] > sequence_record["right"]
            and record["center_x"] < command_right
            and _normalized_text(record["text"])
            and not _is_non_entry_status_text(record["text"])
        ]
        text_records.sort(key=lambda item: (item["center_y"], item["left"]))
        entry_text = inline_text + "".join(
            _normalized_text(record["text"]) for record in text_records
        )
        if not entry_text:
            continue

        key = str(sequence)
        if key in entries:
            raise TicketRecognitionError(
                f"顺序列，第 {key} 项",
                f"同一页重复识别到序号 {key}；"
                f"重复项{_anchor_location(sequence_record)}。",
            )
        entries[key] = entry_text

    return entries


def extract_ticket_data(records, image_shape, source_image=None, rec_model=None):
    """Extract the requested fields from one operation ticket."""
    return {
        "substation": extract_field_value(
            records,
            "单位",
            right_label_text="编号",
            label_anchor=find_unit_anchor(records),
        ),
        "mission": extract_mission(records),
        "id": extract_field_value(records, "编号"),
        "entries": extract_entries(
            records,
            image_shape,
            source_image=source_image,
            rec_model=rec_model,
        ),
    }


def merge_page_entries(entries, page_entries, page_path):
    """Append one page's entries while enforcing document-wide continuity."""
    previous_sequence = int(next(reversed(entries))) if entries else None
    for key, text in page_entries.items():
        sequence = int(key)
        if key in entries:
            raise TicketRecognitionError(
                f"顺序列，第 {key} 项",
                f"序号 {key} 在此前页面已经出现，本页再次出现。"
                "请检查是否重复上传页面、页面顺序错误或序号识别错误。",
                page_path=page_path,
            )
        if previous_sequence is not None and sequence != previous_sequence + 1:
            expected = previous_sequence + 1
            if sequence > expected:
                missing = str(expected) if sequence == expected + 1 else f"{expected}～{sequence - 1}"
                detail = (
                    f"上一有效项序号为 {previous_sequence}，应接续 {expected}，"
                    f"本页实际识别到 {sequence}；未找到第 {missing} 项的有效序号或操作内容。"
                    "请检查是否漏页，或这些项的序号/操作内容不清晰。"
                )
            else:
                detail = (
                    f"上一有效项序号为 {previous_sequence}，应接续 {expected}，"
                    f"本页实际识别到 {sequence}，序号发生倒退。"
                    "请检查页面顺序或序号识别结果。"
                )
            raise TicketRecognitionError(
                f"顺序列，第 {key} 项前的衔接处", detail, page_path=page_path,
            )
        entries[key] = text
        previous_sequence = sequence


def merge_ticket_data_page(ticket_data_list, page_data, page_path):
    """Group one parsed page by mission and enforce upload order."""
    page_entries = page_data["entries"]
    if not page_entries:
        raise TicketRecognitionError(
            "操作项目列（顺序列右侧）",
            "本页未提取到任何有效操作项目。已识别到序号，但对应行右侧的操作内容为空或未被识别。",
            page_path=page_path,
        )

    first_sequence = int(next(iter(page_entries)))
    starts_new_ticket = (
        not ticket_data_list
        or page_data["mission"] != ticket_data_list[-1]["mission"]
    )

    if starts_new_ticket:
        if first_sequence != 1:
            detail = (
                f"本页被判定为新操作票，首个有效操作项应为 1，实际为 {first_sequence}。"
            )
            if not ticket_data_list:
                detail += "请检查是否缺少首页，或首页第 1 项的序号/内容未识别到。"
            else:
                detail += (
                    f"本页操作任务“{page_data['mission']}”与上一票操作任务"
                    f"“{ticket_data_list[-1]['mission']}”不一致。"
                    "如果本页是续页，请检查操作任务区域是否识别错误；"
                    "如果是另一张票，请检查其首页是否缺失或页面顺序是否正确。"
                )
            raise TicketRecognitionError(
                "首个操作项 / 操作任务区域", detail, page_path=page_path,
            )
        new_ticket = {
            "substation": page_data["substation"],
            "mission": page_data["mission"],
            "id": page_data["id"],
            "entries": {},
        }
        merge_page_entries(new_ticket["entries"], page_entries, page_path)
        ticket_data_list.append(new_ticket)
        return

    merge_page_entries(
        ticket_data_list[-1]["entries"],
        page_entries,
        page_path,
    )


def merge_ticket_page(
    ticket_data_list,
    records,
    image_shape,
    page_path,
    source_image=None,
    rec_model=None,
):
    """Parse and merge one page into the ordered operation-ticket list."""
    try:
        page_data = extract_ticket_data(
            records,
            image_shape,
            source_image=source_image,
            rec_model=rec_model,
        )
    except TicketRecognitionError as error:
        raise TicketRecognitionError(
            error.region, error.detail, page_path=page_path,
        ) from error
    merge_ticket_data_page(ticket_data_list, page_data, page_path)

    return ticket_data_list


def _safe_filename_component(value):
    component = INVALID_FILENAME_CHARS.sub("_", str(value)).strip().rstrip(".")
    if not component:
        raise ValueError("Operation ticket filename component cannot be empty")
    return component


def save_ticket_data(ticket_data, output_dir=COMMAND_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    substation = _safe_filename_component(ticket_data["substation"])
    mission = _safe_filename_component(ticket_data["mission"])
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    output_path = output_dir / f"{substation}_{mission}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(ticket_data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    return output_path


def clear_intermediate_outputs(inter_dir=INTER_DIR):
    """Remove artifacts from the previous run and recreate the inter directory."""
    inter_dir = Path(inter_dir).resolve()
    expected_inter_dir = INTER_DIR.resolve()
    if inter_dir != expected_inter_dir:
        raise ValueError(
            f"Refusing to clear unexpected intermediate directory: {inter_dir}"
        )
    if inter_dir.exists():
        shutil.rmtree(inter_dir)
    inter_dir.mkdir(parents=True, exist_ok=True)


def main():
    from paddleocr import TextDetection, TextRecognition

    engine_config = {
        "device_type": "gpu",
        "cpu_threads": 4,
        "run_mode": "mkldnn",
    }
    det_model = TextDetection(
        model_name="PP-OCRv6_medium_det",
        model_dir=DET_MODEL_DIR.resolve(),
        engine="paddle_static",
        engine_config=engine_config,
    )
    rec_model = TextRecognition(
        model_name="PP-OCRv6_medium_rec",
        model_dir=REC_MODEL_DIR.resolve(),
        engine="paddle_static",
        engine_config=engine_config,
    )

    clear_intermediate_outputs()
    PENDING_IMAGES[:] = preprocess_input(
        INPUT_IMAGE,
        det_model,
        rec_model,
        rec_batch_size=REC_BATCH_SIZE,
    )

    ticket_data_list = []
    try:
        for page_path in PENDING_IMAGES:
            for detection_result in det_model.predict(str(page_path)):
                detection_result.print()
                crop_paths = save_text_regions(
                    detection_result,
                    TEXT_CROP_DIR,
                    source_path=page_path,
                )
                detection_result.save_to_img(save_path=str(INTER_DIR))

                records = recognize_text_regions(
                    rec_model,
                    crop_paths,
                    detection_result["dt_polys"],
                    detection_result["dt_scores"],
                )
                json_path, annotated_path = save_recognition_results(
                    page_path,
                    detection_result["input_img"],
                    records,
                )
                print(f"Saved per-region OCR to {json_path} and {annotated_path}")
                merge_ticket_page(
                    ticket_data_list,
                    records,
                    detection_result["input_img"].shape,
                    page_path,
                    source_image=detection_result["input_img"],
                    rec_model=rec_model,
                )
    except TicketRecognitionError as error:
        print(str(error))
        print(f"请查看本页逐框识别结果：{json_path}\n标注图：{annotated_path}")
        return

    if not ticket_data_list:
        raise ValueError("No operation-ticket pages produced OCR results")

    for ticket_data in ticket_data_list:
        output_path = save_ticket_data(ticket_data)
        print(json.dumps(ticket_data, ensure_ascii=False, indent=2))
        print(f"Saved operation ticket data to {output_path}")


if __name__ == "__main__":
    main()
