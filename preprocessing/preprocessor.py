import logging
import re

from bs4 import BeautifulSoup

from core.interfaces import TextPreprocessor

logger = logging.getLogger(__name__)


class NLPPreprocessor(TextPreprocessor):
    """Lightweight preprocessing for embedding-based classification."""

    def __init__(self) -> None:
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
        self._whitespace_pattern = re.compile(r"\s+")

    def preprocess(self, subject: str, body_html: str, body_plain: str) -> str:
        body = self.strip_html(body_html) if body_html else body_plain

        text = f"{subject} {body}"

        text = self._url_pattern.sub("", text)

        text = self._whitespace_pattern.sub(" ", text).strip()

        logger.debug(f"Preprocessed: {len(text)} chars")
        return text

    @staticmethod
    def strip_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
