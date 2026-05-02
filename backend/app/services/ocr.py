from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def ocr_pdf_page(file_path: Path, page_index: int) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""

    try:
        images = convert_from_path(
            str(file_path),
            first_page=page_index + 1,
            last_page=page_index + 1,
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0])
    except Exception as exc:
        logger.warning("OCR failed for page %s: %s", page_index + 1, exc)
        return ""
