from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PAGE_ROOT = PROJECT_ROOT / "inter" / "preprocessed"
IMAGE_INPUT_ROOT = PROJECT_ROOT / "assets" / "imgs"
PDF_DPI = 260
HEADER_HEIGHT_RATIO = 0.12
TICKET_TITLE = "变电站倒闸操作票"
TITLE_MIN_SIMILARITY = 0.6
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 0.5


def _normalized_text(text):
    return "".join(str(text).split())


def _read_image(image_path):
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def _is_path_within(path, directory):
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:
        return False
    return True


def _enhance_document_image(image):
    """Improve local contrast and edge clarity while preserving image colors."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE,
    )
    enhanced_lightness = clahe.apply(lightness)
    contrast_enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, channel_a, channel_b)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(
        contrast_enhanced,
        (0, 0),
        sigmaX=SHARPEN_SIGMA,
        sigmaY=SHARPEN_SIGMA,
    )
    return cv2.addWeighted(
        contrast_enhanced,
        1.0 + SHARPEN_AMOUNT,
        blurred,
        -SHARPEN_AMOUNT,
        0,
    )


def _write_png(image_path, image):
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise OSError(f"Failed to encode enhanced PNG image: {image_path}")
    encoded.tofile(str(image_path))


def _enhance_input_image(image_path, output_root=None):
    image_path = Path(image_path)
    if output_root is None:
        output_root = PDF_PAGE_ROOT
    relative_path = image_path.resolve().relative_to(IMAGE_INPUT_ROOT.resolve())
    output_path = (
        Path(output_root)
        / relative_path.parent
        / f"{image_path.stem}_enhanced.png"
    )
    _write_png(output_path, _enhance_document_image(_read_image(image_path)))
    return output_path


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
    """Render/filter one PDF or supported image and return ticket page paths."""
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        candidate_pages = _render_pdf_pages(input_path)
    elif suffix in IMAGE_SUFFIXES:
        candidate_pages = [
            _enhance_input_image(input_path)
            if _is_path_within(input_path, IMAGE_INPUT_ROOT)
            else input_path
        ]
    else:
        raise ValueError(
            f"Unsupported input type {input_path.suffix!r}; "
            "expected .pdf, .png, .jpg, or .jpeg"
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
