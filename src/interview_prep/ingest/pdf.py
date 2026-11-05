from io import BytesIO

from pypdf import PdfReader


def extract_text(pdf_bytes: bytes) -> str:
    """Plain text of a PDF, pages joined by blank lines. Raises ValueError if it has no text layer."""
    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    if not text:
        raise ValueError("No text found in the PDF. It may be a scanned image; export it as a text PDF.")
    return text
