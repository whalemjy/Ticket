import json
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from preprocess import preprocess_input

PROJECT_ROOT = Path(__file__).resolve().parent
# TODO: Replace this directory after the storage layout is decided.
MODEL_ROOT = PROJECT_ROOT / "models"
DET_MODEL_DIR = MODEL_ROOT / "PP-OCRv6_small_det"
REC_MODEL_DIR = MODEL_ROOT / "PP-OCRv6_small_rec"
INTER_DIR = PROJECT_ROOT / "inter"
TEXT_CROP_DIR = INTER_DIR / "text_crops"
COMMAND_DIR = Path("./command")
INPUT_IMAGE = Path("./assets/pdfs/110kV夺锦变电站.pdf")
PENDING_IMAGES = []

# The sequence numbers are in the same narrow column as the "顺序" header.
SEQUENCE_X_TOLERANCE_RATIO = 0.06
# Ignore the check-mark column on the right side of the operation table.
COMMAND_RIGHT_BOUNDARY_RATIO = 0.89
REC_BATCH_SIZE = 4
OPERATION_TASK_MIN_SIMILARITY = 0.8
OPERATION_TASK_MIN_CONFIDENCE = 0.75
UPLOAD_ORDER_ERROR = "请按照正确顺序上传操作票或拍摄更清晰的操作票"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


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

    # Recognition models generally work better when tall text is horizontal.
    if crop_height / crop_width >= 1.5:
        crop = np.rot90(crop).copy()
    return crop


def _write_png(image_path, image):
    """Write a PNG through NumPy so Windows Unicode paths remain intact."""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed to encode PNG image: {image_path}")
    encoded.tofile(image_path)


def save_text_regions(result, output_dir, source_path):
    """Save every detected text region and return the generated file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image = result["input_img"].copy()
    source_name = Path(source_path).stem
    saved_paths = []

    for region_index, points in enumerate(result["dt_polys"]):
        crop = crop_text_region(source_image, points)
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


def _normalized_text(text):
    return re.sub(r"\s+", "", text)


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
        raise ValueError(f"Cannot find the {target_text!r} text anchor")
    if prefer_confidence:
        return max(
            candidates,
            key=lambda item: (item[1]["score"] or 0.0, item[0]),
        )[1]
    return max(
        candidates,
        key=lambda item: (item[0], item[1]["score"] or 0.0),
    )[1]


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
        raise ValueError("Cannot find a value to the right of '操作任务'")
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
        raise ValueError(f"Cannot find a value to the right of {label_text!r}")
    return value


def _sequence_value(text):
    normalized = _normalized_text(text)
    if re.fullmatch(r"\d{1,3}", normalized):
        return int(normalized)
    return None


def _split_merged_sequence_entry(record, sequence_anchor):
    """Split a sequence prefix only when its box crosses the sequence column."""
    normalized = _normalized_text(record["text"])
    match = re.fullmatch(r"(\d{1,3})(\D.*)", normalized)
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


def extract_entries(records, image_shape):
    """Extract operation item text by anchoring rows on their sequence numbers."""
    image_height, image_width = image_shape[:2]
    sequence_anchor = find_text_anchor(records, "顺序")
    x_tolerance = image_width * SEQUENCE_X_TOLERANCE_RATIO

    sequence_records = []
    for record in records:
        sequence = _sequence_value(record["text"])
        is_below_anchor = record["center_y"] > sequence_anchor["bottom"]
        is_in_sequence_column = (
            abs(record["center_x"] - sequence_anchor["center_x"]) <= x_tolerance
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

    sequence_records.sort(key=lambda item: item[1]["center_y"])
    if not sequence_records:
        raise ValueError("No sequence numbers were found below the '顺序' anchor")

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
        ]
        text_records.sort(key=lambda item: (item["center_y"], item["left"]))
        entry_text = inline_text + "".join(
            _normalized_text(record["text"]) for record in text_records
        )
        if not entry_text:
            continue

        key = str(sequence)
        if key in entries:
            raise ValueError(f"Duplicate sequence number detected: {key}")
        entries[key] = entry_text

    return entries


def extract_ticket_data(records, image_shape):
    """Extract the requested fields from one operation ticket."""
    return {
        "substation": extract_field_value(
            records,
            "单位",
            right_label_text="编号",
        ),
        "mission": extract_mission(records),
        "id": extract_field_value(records, "编号"),
        "entries": extract_entries(records, image_shape),
    }


def merge_page_entries(entries, page_entries, page_path):
    """Append one page's entries while enforcing document-wide continuity."""
    previous_sequence = int(next(reversed(entries))) if entries else None
    for key, text in page_entries.items():
        sequence = int(key)
        if key in entries:
            raise ValueError(UPLOAD_ORDER_ERROR)
        if previous_sequence is not None and sequence != previous_sequence + 1:
            raise ValueError(UPLOAD_ORDER_ERROR)
        entries[key] = text
        previous_sequence = sequence


def merge_ticket_data_page(ticket_data_list, page_data, page_path):
    """Group one parsed page by mission and enforce upload order."""
    page_entries = page_data["entries"]
    if not page_entries:
        raise ValueError(UPLOAD_ORDER_ERROR)

    first_sequence = int(next(iter(page_entries)))
    starts_new_ticket = (
        not ticket_data_list
        or page_data["mission"] != ticket_data_list[-1]["mission"]
    )

    if starts_new_ticket:
        if first_sequence != 1:
            raise ValueError(UPLOAD_ORDER_ERROR)
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


def merge_ticket_page(ticket_data_list, records, image_shape, page_path):
    """Parse and merge one page into the ordered operation-ticket list."""
    page_data = extract_ticket_data(records, image_shape)
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
        model_name="PP-OCRv6_small_det",
        model_dir=DET_MODEL_DIR.resolve(),
        engine="paddle_static",
        engine_config=engine_config,
    )
    rec_model = TextRecognition(
        model_name="PP-OCRv6_small_rec",
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
                merge_ticket_page(
                    ticket_data_list,
                    records,
                    detection_result["input_img"].shape,
                    page_path,
                )
    except ValueError as error:
        if str(error) == UPLOAD_ORDER_ERROR:
            print(UPLOAD_ORDER_ERROR)
            return
        raise

    if not ticket_data_list:
        raise ValueError("No operation-ticket pages produced OCR results")

    for ticket_data in ticket_data_list:
        output_path = save_ticket_data(ticket_data)
        print(json.dumps(ticket_data, ensure_ascii=False, indent=2))
        print(f"Saved operation ticket data to {output_path}")


if __name__ == "__main__":
    main()
