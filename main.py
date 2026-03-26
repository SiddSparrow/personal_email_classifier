"""
Email classification pipeline orchestrator.
Composes strategies via dependency injection and runs the full pipeline.
"""

import logging
import sys

from core.interfaces import (
    EmailReader,
    TextPreprocessor,
    EmailClassifier,
    Notifier,
    StateManager,
    ProcessedEmail,
)
from core.config import load_config, setup_logging, AppConfig
from reader.gmail_reader import GmailReader
from preprocessing.preprocessor import NLPPreprocessor
from classification.classifier import LogisticRegressionClassifier
from notification.notifier import TelegramNotifier
from state.state_manager import JsonStateManager

logger = logging.getLogger(__name__)

PREVIEW_LENGTH = 300


class EmailPipeline:
    """
    Orchestrates the full email classification pipeline.
    Depends ONLY on abstractions (interfaces), not concrete implementations.
    """

    def __init__(
        self,
        reader: EmailReader,
        preprocessor: TextPreprocessor,
        classifier: EmailClassifier,
        notifier: Notifier,
        state: StateManager,
        max_emails: int = 50,
        mark_as_read: bool = False,
    ) -> None:
        self._reader = reader
        self._preprocessor = preprocessor
        self._classifier = classifier
        self._notifier = notifier
        self._state = state
        self._max_emails = max_emails
        self._mark_as_read = mark_as_read

    def run(self) -> dict:
        """
        Execute the full pipeline:
        1. Load state → 2. Fetch emails → 3. Filter processed →
        4. Preprocess → 5. Classify → 6. Notify → 7. Save state
        """
        stats = {
            "fetched": 0,
            "skipped": 0,
            "classified": 0,
            "notified": 0,
            "errors": 0,
        }

        self._state.load()

        emails = self._reader.fetch_unread(self._max_emails)
        stats["fetched"] = len(emails)

        for email in emails:
            if self._state.is_processed(email.message_id):
                stats["skipped"] += 1
                logger.debug(f"Skipping already-processed: {email.message_id}")
                continue

            try:
                cleaned_text = self._preprocessor.preprocess(
                    email.subject, email.body_html, email.body_plain
                )

                result = self._classifier.classify(cleaned_text)
                stats["classified"] += 1
                logger.info(
                    f"Email {email.message_id}: {result.category} "
                    f"({result.confidence:.1%})"
                )

                if result.is_positive:
                    preview_source = email.body_plain or self._preprocessor.strip_html(email.body_html)
                    preview = preview_source[:PREVIEW_LENGTH].strip()
                    if len(preview_source) > PREVIEW_LENGTH:
                        preview += "..."

                    processed = ProcessedEmail(
                        email=email,
                        classification=result,
                        preview=preview,
                    )

                    if self._notifier.notify(processed):
                        stats["notified"] += 1

                self._state.mark_processed(email.message_id)

                if self._mark_as_read:
                    self._reader.mark_as_read(email.message_id)

            except Exception:
                stats["errors"] += 1
                logger.exception(f"Error processing email {email.message_id}")

        self._state.save()
        logger.info(f"Pipeline complete: {stats}")
        return stats


def build_pipeline(config: AppConfig) -> EmailPipeline:
    """
    Composition Root: wires concrete implementations to the pipeline.
    This is the ONLY place that knows about both abstractions AND concretes.
    """
    reader = GmailReader(
        credentials_file=config.gmail_credentials_file,
        token_file=config.gmail_token_file,
        query=config.gmail_query,
    )
    preprocessor = NLPPreprocessor()
    classifier = LogisticRegressionClassifier(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
    )
    notifier = TelegramNotifier(
        bot_token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
    )
    state = JsonStateManager(state_file=config.state_file)

    return EmailPipeline(
        reader=reader,
        preprocessor=preprocessor,
        classifier=classifier,
        notifier=notifier,
        state=state,
        max_emails=config.max_emails_per_run,
        mark_as_read=config.mark_as_read,
    )


def main() -> None:
    try:
        config = load_config()
        setup_logging(config)
        logger.info("Starting email classification pipeline")

        pipeline = build_pipeline(config)
        stats = pipeline.run()

        if stats["errors"] > 0:
            sys.exit(1)

    except Exception:
        logging.exception("Fatal error in pipeline")
        sys.exit(2)


if __name__ == "__main__":
    main()
