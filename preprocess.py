from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PAGE_ROOT = PROJECT_ROOT / "inter" / "preprocessed"
PDF_DPI = 260
HEADER_HEIGHT_RATIO = 0.12
TICKET_TITLE = "变电站倒闸操作票"
TITLE_MIN_SIMILARITY = 0.6


def _normalized_text(text):
    return "".join(str(text).split())


def _read_image(image_path):
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def _crop_text_region(image, points):
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
    return cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _render_pdf_pages(pdf_path, output_root=PDF_PAGE_ROOT):
    try:
        import pypdfium2 as pdfium
    except ImportError as error:
        raise RuntimeError(
            "PDF input requires pypdfium2 in the active Python environment"
        ) from error

    output_dir = output_root / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    page_paths = []
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            bitmap = None
            image = None
            try:
                bitmap = page.render(scale=PDF_DPI / 72)
                image = bitmap.to_pil()
                page_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1:04d}.png"
                image.save(page_path, format="PNG")
                page_paths.append(page_path)
            finally:
                if image is not None:
                    image.close()
                if bitmap is not None:
                    bitmap.close()
                page.close()
    finally:
        document.close()
    return page_paths


def _matches_ticket_title(text):
    normalized = _normalized_text(text)
    if TICKET_TITLE in normalized:
        return True
    return (
        SequenceMatcher(None, TICKET_TITLE, normalized).ratio()
        >= TITLE_MIN_SIMILARITY
    )


def is_operation_ticket_page(image_path, det_model, rec_model, rec_batch_size=4):
    """Return whether the top band contains the operation-ticket title."""
    image = _read_image(image_path)
    header_height = max(int(round(image.shape[0] * HEADER_HEIGHT_RATIO)), 1)
    header_image = image[:header_height]
    recognized_regions = []

    for detection_result in det_model.predict(header_image):
        polygons = list(detection_result["dt_polys"])
        if not polygons:
            continue
        crops = [_crop_text_region(header_image, points) for points in polygons]
        predictions = rec_model.predict(crops, batch_size=rec_batch_size)
        if len(predictions) != len(polygons):
            raise ValueError(
                "Header recognition result and polygon counts differ: "
                f"{len(predictions)} != {len(polygons)}"
            )

        for prediction, points in zip(predictions, polygons):
            points = np.asarray(points)
            recognized_regions.append(
                (
                    float(points[:, 1].mean()),
                    float(points[:, 0].min()),
                    str(prediction["rec_text"]),
                )
            )

    recognized_regions.sort(key=lambda item: (item[0], item[1]))
    texts = [item[2] for item in recognized_regions]
    return any(_matches_ticket_title(text) for text in texts) or (
        TICKET_TITLE in _normalized_text("".join(texts))
    )


def preprocess_input(input_path, det_model, rec_model, rec_batch_size=4):
    """Render/filter one PDF or PNG and return operation-ticket page paths."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        candidate_pages = _render_pdf_pages(input_path)
    elif suffix == ".png":
        candidate_pages = [input_path]
    else:
        raise ValueError(
            f"Unsupported input type {input_path.suffix!r}; expected .pdf or .png"
        )

    pending_images = [
        page_path
        for page_path in candidate_pages
        if is_operation_ticket_page(
            page_path,
            det_model,
            rec_model,
            rec_batch_size=rec_batch_size,
        )
    ]
    if not pending_images:
        raise ValueError(
            f"No pages containing the title {TICKET_TITLE!r} were found in {input_path}"
        )
    return pending_images
