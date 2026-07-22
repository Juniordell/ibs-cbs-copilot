from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF, imported as pymupdf in v1.24+

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cleanup patterns
# ---------------------------------------------------------------------------

# Headers/footers from planalto.gov.br PDFs we want to strip
NOISE_PATTERNS = [
    re.compile(r"Presidência da República", re.IGNORECASE),
    re.compile(r"Casa Civil", re.IGNORECASE),
    re.compile(r"Secretaria Especial para Assuntos Jurídicos", re.IGNORECASE),
    re.compile(r"Este texto não substitui.*", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # standalone page numbers
    re.compile(r"planalto\.gov\.br", re.IGNORECASE),
]

# Structural markers we want to preserve on their own line so the chunker
# can split cleanly. Order matters: match the most specific first.
STRUCTURAL_MARKERS = [
    re.compile(r"(Art\.\s*\d+[º°]?\s*[-.]?)"),      # Art. 12, Art. 12º
    re.compile(r"(§\s*\d+[º°]?)"),                   # § 1º, § 2
    re.compile(r"(Parágrafo único\.?)", re.IGNORECASE),
    re.compile(r"(CAPÍTULO\s+[IVXLCDM]+)", re.IGNORECASE),
    re.compile(r"(SEÇÃO\s+[IVXLCDM]+)", re.IGNORECASE),
    re.compile(r"(TÍTULO\s+[IVXLCDM]+)", re.IGNORECASE),
    re.compile(r"(LIVRO\s+[IVXLCDM]+)", re.IGNORECASE),
]


def is_noise(line: str) -> bool:
    """Return True if the line is a header, footer, or page-number artifact."""
    stripped = line.strip()
    if not stripped:
        return False  # blank lines handled elsewhere
    return any(pattern.search(stripped) for pattern in NOISE_PATTERNS)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace but keep paragraph breaks."""
    # Preserve double newlines (paragraph breaks), collapse single ones
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def enforce_structural_breaks(text: str) -> str:
    """Ensure every article/paragraph marker starts on its own line.

    PyMuPDF sometimes joins these markers with surrounding text. The chunker
    depends on them being at line-start, so we force it here.
    """
    for pattern in STRUCTURAL_MARKERS:
        # Insert a newline before the marker if it's not already there
        text = pattern.sub(r"\n\1", text)
    # Clean up any triple newlines we may have introduced
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract raw text from a PDF, one page at a time."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Opening %s", pdf_path)
    pages: list[str] = []

    with pymupdf.open(pdf_path) as doc:
        logger.info("PDF has %d pages", len(doc))
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text")
            pages.append(page_text)
            if page_num % 50 == 0:
                logger.info("Processed %d pages", page_num)

    return "\n".join(pages)


def clean_text(raw: str) -> str:
    """Drop noise lines, normalize whitespace, enforce structural breaks."""
    lines = raw.splitlines()
    kept = [line for line in lines if not is_noise(line)]
    text = "\n".join(kept)
    text = normalize_whitespace(text)
    text = enforce_structural_breaks(text)
    return text


def pdf_to_text(pdf_path: Path, output_path: Path) -> Path:
    """Full pipeline: extract, clean, save."""
    raw = extract_pdf_text(pdf_path)
    cleaned = clean_text(raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    logger.info(
        "Wrote %d characters to %s (from %d raw)",
        len(cleaned), output_path, len(raw),
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from a legal PDF")
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the PDF file")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path where the .txt will be written")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        pdf_to_text(args.input, args.output)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1
    except Exception:
        logger.exception("Extraction failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())