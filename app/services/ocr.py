"""OCR utilities for processing PDF documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from loguru import logger
import numpy as np
import pypdfium2 as pdfium
from rapidocr import RapidOCR
import os
import platform


class OCRExtractionError(RuntimeError):
    """Raised when OCR processing fails."""


@dataclass
class OCRResult:
    """Details of a completed OCR extraction."""

    text: str
    page_count: int
    output_path: Path


def extract_text_from_pdf(
    pdf_bytes: bytes,
    output_path: Path,
    *,
    language: str = "eng",
    dpi: int = 300,
) -> OCRResult:
    """Run OCR on a PDF file and persist the extracted text.

    Args:
        pdf_bytes: Raw bytes of the PDF document.
        output_path: Destination path for the extracted text file.
        language: Language hint for Tesseract. Defaults to English.
        dpi: Rendering resolution for each PDF page.

    Returns:
        The extracted text content.

    Raises:
        OCRExtractionError: If the document cannot be processed.
    """

    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
    except Exception as exc:  # pragma: no cover - defensive catch
        raise OCRExtractionError("Unable to open PDF document") from exc

    if len(pdf) == 0:
        raise OCRExtractionError("PDF did not contain any pages")

    # Initialize OCR engine once with English + local font to avoid remote downloads
    def _default_font_path() -> str:
        # Allow override via env
        env_font = os.getenv("OCR_FONT_PATH", "").strip()
        if env_font and os.path.exists(env_font):
            return env_font
        system = platform.system()
        candidates = []
        if system == "Windows":
            candidates = [
                "C:\\Windows\\Fonts\\arial.ttf",
                "C:\\Windows\\Fonts\\ARIAL.TTF",
                "C:\\Windows\\Fonts\\calibri.ttf",
            ]
        elif system == "Darwin":
            candidates = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial.ttf",
            ]
        else:
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return ""

    font_path = _default_font_path()

    # Hard-stop any remote font download attempts by overriding RapidOCR's font resolver
    try:
        from rapidocr.utils import vis_res as _rapid_vis

        def _patched_get_font_path(self, given_font_path, lang_type):  # type: ignore[no-redef]
            # Prefer provided path; otherwise pick a local system font
            if given_font_path and os.path.exists(given_font_path):
                return given_font_path
            fallback = _default_font_path()
            return fallback if fallback else given_font_path

        _rapid_vis.VisRes.get_font_path = _patched_get_font_path  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        # Force English to avoid Chinese font fetch; provide local font if found
        ocr_engine = (
            RapidOCR(lang_type="en", font_path=font_path)
            if font_path
            else RapidOCR(lang_type="en")
        )
    except TypeError:
        # Older versions may not accept font_path kw; attempt to set via cfg
        ocr_engine = RapidOCR()
        try:
            if hasattr(ocr_engine, "cfg"):
                try:
                    ocr_engine.cfg.lang_type = "en"  # type: ignore[attr-defined]
                except Exception:
                    pass
                if font_path:
                    ocr_engine.cfg.font_path = font_path  # type: ignore[attr-defined]
        except Exception:
            pass

    scale = dpi / 72.0
    pages_text: List[str] = []
    for index, page in enumerate(pdf, start=1):
        try:
            pil_image = page.render(scale=scale).to_pil()
        except Exception as exc:  # pragma: no cover - defensive catch
            raise OCRExtractionError(f"Failed to render page {index}") from exc

        logger.debug("Running OCR on page {page}", page=index)

        # RapidOCR accepts ndarray/PIL/path; we provide ndarray
        image_np = np.array(pil_image)
        ocr_output = ocr_engine(image_np)

        # Normalize outputs across RapidOCR versions — collect TEXT ONLY
        page_text_chunks: List[str] = []
        try:
            # Newer API: RapidOCROutput may expose texts/txts/result
            if hasattr(ocr_output, "texts") and isinstance(ocr_output.texts, (list, tuple)):  # type: ignore[attr-defined]
                for text in ocr_output.texts:  # type: ignore[attr-defined]
                    if text:
                        page_text_chunks.append(str(text).strip())
            elif hasattr(ocr_output, "txts") and isinstance(ocr_output.txts, (list, tuple)):  # type: ignore[attr-defined]
                for text in ocr_output.txts:  # type: ignore[attr-defined]
                    if text:
                        page_text_chunks.append(str(text).strip())
            elif hasattr(ocr_output, "result") and isinstance(ocr_output.result, list):  # type: ignore[attr-defined]
                for item in ocr_output.result:  # type: ignore[attr-defined]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text = item[1]
                        if text:
                            page_text_chunks.append(str(text).strip())
                    elif isinstance(item, dict) and "text" in item:
                        text = item.get("text")
                        if text:
                            page_text_chunks.append(str(text).strip())
            elif isinstance(ocr_output, tuple) and len(ocr_output) >= 1:
                result_list = ocr_output[0]
                for item in (result_list or []):
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text = item[1]
                        if text:
                            page_text_chunks.append(str(text).strip())
                    elif isinstance(item, dict) and "text" in item:
                        text = item.get("text")
                        if text:
                            page_text_chunks.append(str(text).strip())
        except Exception:
            # If parsing fails, skip non-text payloads (do not dump objects)
            pass

        pages_text.append("\n".join(page_text_chunks).strip())

    full_text = "\n\n".join(filter(None, pages_text)).strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(full_text, encoding="utf-8")

    page_count = len(pages_text)
    logger.info(
        "OCR extraction complete",
        page_count=page_count,
        output=str(output_path),
    )

    return OCRResult(text=full_text, page_count=page_count, output_path=output_path)


