import logging
import re

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

from core.interfaces import TextPreprocessor

logger = logging.getLogger(__name__)


class NLPPreprocessor(TextPreprocessor):
    """NLP preprocessing pipeline: HTML strip, tokenize, stopwords, stemming."""

    def __init__(
        self,
        use_stemmer: bool = True,
        languages: tuple[str, ...] = ("portuguese", "english"),
    ) -> None:
        self._use_stemmer = use_stemmer

        # Ensure NLTK resources are available
        for resource in ("stopwords", "punkt", "punkt_tab", "rslp"):
            try:
                nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else resource)
            except LookupError:
                nltk.download(resource, quiet=True)

        self._stopwords: set[str] = set()
        for lang in languages:
            self._stopwords.update(stopwords.words(lang))

        self._stemmer = RSLPStemmer() if use_stemmer else None
        self._url_pattern = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\S+")
        self._punct_pattern = re.compile(
            r"[^a-záàâãéèêíïóôõöúüçñ0-9\s]", re.IGNORECASE
        )

    def preprocess(self, subject: str, body_html: str, body_plain: str) -> str:
        # 1. Extract text from HTML (fallback to plain)
        body = self.strip_html(body_html) if body_html else body_plain

        # 2. Merge subject + body
        text = f"{subject} {body}"

        # 3. Lowercase
        text = text.lower()

        # 4. Remove URLs and email addresses
        text = self._url_pattern.sub("", text)

        # 5. Remove punctuation (keep letters, numbers, spaces, PT accented chars)
        text = self._punct_pattern.sub(" ", text)

        # 6. Tokenize
        tokens = word_tokenize(text)

        # 7. Remove stopwords and single-char tokens
        tokens = [t for t in tokens if t not in self._stopwords and len(t) > 1]

        # 8. Optional stemming
        if self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        # 9. Rejoin
        result = " ".join(tokens)
        logger.debug(f"Preprocessed: {len(tokens)} tokens, {len(result)} chars")
        return result

    @staticmethod
    def strip_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)
