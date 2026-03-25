from abc import ABC, abstractmethod
from dataclasses import dataclass


# ── Value Objects ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmailMessage:
    """Immutable representation of a fetched email."""
    message_id: str
    subject: str
    sender_name: str
    sender_email: str
    date: str
    body_html: str
    body_plain: str
    gmail_link: str


@dataclass(frozen=True)
class ClassificationResult:
    """Immutable result from the classifier."""
    category: str
    confidence: float
    is_positive: bool


@dataclass(frozen=True)
class ProcessedEmail:
    """An email bundled with its classification, ready for notification."""
    email: EmailMessage
    classification: ClassificationResult
    preview: str


# ── Strategy Interfaces ───────────────────────────────────────────────────────

class EmailReader(ABC):
    """Reads emails from a source. Swap for IMAP, Outlook, etc."""

    @abstractmethod
    def fetch_unread(self, max_results: int) -> list[EmailMessage]:
        ...

    @abstractmethod
    def mark_as_read(self, message_id: str) -> None:
        ...


class TextPreprocessor(ABC):
    """Cleans and normalizes raw email content for classification."""

    @abstractmethod
    def preprocess(self, subject: str, body_html: str, body_plain: str) -> str:
        ...

    @abstractmethod
    def strip_html(self, html: str) -> str:
        ...


class EmailClassifier(ABC):
    """Classifies preprocessed text. Swap for SVM, transformer, etc."""

    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        ...


class Notifier(ABC):
    """Sends notifications. Swap for Slack, Discord, email, etc."""

    @abstractmethod
    def notify(self, processed_email: ProcessedEmail) -> bool:
        ...


class StateManager(ABC):
    """Tracks which emails have been processed."""

    @abstractmethod
    def is_processed(self, message_id: str) -> bool:
        ...

    @abstractmethod
    def mark_processed(self, message_id: str) -> None:
        ...

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...
