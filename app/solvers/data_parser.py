"""Data parsing utilities able to ingest multiple document types."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
from bs4 import BeautifulSoup

from app.utils.logger import configure_logger, log_extra

logger = configure_logger(__name__)

try:  # pragma: no cover - optional dependency
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:  # pragma: no cover
    import PyPDF2
except ModuleNotFoundError:  # pragma: no cover
    PyPDF2 = None  # type: ignore

try:  # pragma: no cover
    import pytesseract
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


@dataclass
class ParsedDocument:
    """Container storing parsed text and optional structured artefacts."""

    text: str
    tables: List[pd.DataFrame]
    metadata: Dict[str, Any]


class DataParser:
    """High level parser providing resilient fallbacks for tricky formats."""

    async def parse(self, data: Union[str, bytes], *, format_hint: Optional[str] = None) -> ParsedDocument:
        if format_hint is None:
            format_hint = self._infer_format(data)
        logger.info("Parsing payload", extra=log_extra(format=format_hint))
        if format_hint == "json":
            parsed = self._parse_json(data)
            text = json.dumps(parsed, indent=2)
            return ParsedDocument(text=text, tables=[], metadata={})
        if format_hint == "csv":
            df = self._parse_csv(data)
            return ParsedDocument(text=df.to_csv(index=False), tables=[df], metadata={})
        if format_hint == "html":
            return self._parse_html(data)
        if format_hint == "pdf":
            return self._parse_pdf(data)
        if format_hint == "image":
            return self._parse_image(data)
        return ParsedDocument(text=self._ensure_text(data), tables=[], metadata={})

    # ------------------------------------------------------------------
    def _infer_format(self, data: Union[str, bytes]) -> str:
        if isinstance(data, bytes):
            if data.startswith(b"%PDF"):
                return "pdf"
            if data.startswith(b"\x89PNG") or data[:2] == b"\xff\xd8":
                return "image"
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                return "binary"
        else:
            text = data
        stripped = text.strip().lower()
        if stripped.startswith("<!doctype") or stripped.startswith("<html"):
            return "html"
        if stripped.startswith("{") or stripped.startswith("["):
            return "json"
        if "," in stripped and "\n" in stripped.split(",")[0]:
            return "csv"
        return "text"

    def _ensure_text(self, data: Union[str, bytes]) -> str:
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return data

    # ------------------------------------------------------------------
    def _parse_json(self, data: Union[str, bytes]) -> Any:
        return json.loads(self._ensure_text(data))

    def _parse_csv(self, data: Union[str, bytes]) -> pd.DataFrame:
        text = self._ensure_text(data)
        sample = text[:1024]
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        df = pd.read_csv(io.StringIO(text), delimiter=delimiter)
        return df

    def _parse_html(self, data: Union[str, bytes]) -> ParsedDocument:
        soup = BeautifulSoup(self._ensure_text(data), "html.parser")
        text = soup.get_text("\n", strip=True)
        tables: List[pd.DataFrame] = []
        for table in soup.find_all("table"):
            headers = [cell.get_text(strip=True) for cell in table.find_all("th")]
            rows = [
                [cell.get_text(strip=True) for cell in row.find_all("td")]
                for row in table.find_all("tr")
                if row.find_all("td")
            ]
            if rows:
                df = pd.DataFrame(rows, columns=headers or None)
                tables.append(df)
        metadata = {"title": soup.title.string if soup.title else None}
        return ParsedDocument(text=text, tables=tables, metadata=metadata)

    def _parse_pdf(self, data: bytes) -> ParsedDocument:
        if pdfplumber:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                texts: List[str] = []
                tables: List[pd.DataFrame] = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    texts.append(text)
                    for table in page.extract_tables() or []:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
                metadata = pdf.metadata or {}
                return ParsedDocument("\n".join(texts), tables, metadata)
        if PyPDF2:
            reader = PyPDF2.PdfReader(io.BytesIO(data))
            texts = [page.extract_text() or "" for page in reader.pages]
            return ParsedDocument("\n".join(texts), [], {"pages": len(reader.pages)})
        raise RuntimeError("PDF parsing requires pdfplumber or PyPDF2")

    def _parse_image(self, data: bytes) -> ParsedDocument:
        if not (pytesseract and Image):
            raise RuntimeError("Image parsing requires pytesseract and pillow")
        image = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(image)
        return ParsedDocument(text=text, tables=[], metadata={"size": image.size, "format": image.format})
