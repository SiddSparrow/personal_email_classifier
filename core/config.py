import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration. Validated at construction time."""

    # Telegram
    telegram_bot_token: str
    telegram_chat_id: str

    # Gmail
    gmail_credentials_file: Path
    gmail_token_file: Path
    gmail_query: str

    # Classifier
    confidence_threshold: float
    model_path: Path

    # Behavior
    mark_as_read: bool
    max_emails_per_run: int

    # State
    state_file: Path

    # Logging
    log_level: str
    log_file: Path


def load_config(env_path: str = ".env") -> AppConfig:
    """Load config from .env file. Raises ValueError if required vars are missing."""
    load_dotenv(env_path)

    def _require(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    return AppConfig(
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_require("TELEGRAM_CHAT_ID"),
        gmail_credentials_file=Path(os.getenv("GMAIL_CREDENTIALS_FILE", "./credentials.json")),
        gmail_token_file=Path(os.getenv("GMAIL_TOKEN_FILE", "./credentials/token.json")),
        gmail_query=os.getenv("GMAIL_QUERY", "is:unread newer_than:1d"),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
        model_path=Path(os.getenv("MODEL_PATH", "./models/model.pkl")),
        mark_as_read=os.getenv("MARK_AS_READ", "false").lower() == "true",
        max_emails_per_run=int(os.getenv("MAX_EMAILS_PER_RUN", "50")),
        state_file=Path(os.getenv("STATE_FILE", "./data/state.json")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_file=Path(os.getenv("LOG_FILE", "./logs/classifier.log")),
    )


def setup_logging(config: AppConfig) -> None:
    """Configure root logger with file and stderr handlers."""
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
