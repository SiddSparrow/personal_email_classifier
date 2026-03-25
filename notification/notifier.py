import logging

import requests

from core.interfaces import Notifier, ProcessedEmail

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier(Notifier):
    """Sends formatted notifications via Telegram Bot API using HTTP POST."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = TELEGRAM_API.format(token=bot_token)

    def notify(self, processed_email: ProcessedEmail) -> bool:
        message = self._format_message(processed_email)
        try:
            response = requests.post(
                self._url,
                json={
                    "chat_id": self._chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            response.raise_for_status()
            logger.info(
                f"Telegram notification sent for {processed_email.email.message_id}"
            )
            return True
        except Exception:
            logger.exception("Failed to send Telegram notification")
            return False

    @staticmethod
    def _format_message(pe: ProcessedEmail) -> str:
        e = pe.email
        c = pe.classification
        return (
            "🚨 <b>Nova Oferta de Emprego Detectada!</b>\n\n"
            f"📌 <b>Assunto:</b> {e.subject}\n"
            f"👤 <b>De:</b> {e.sender_name} &lt;{e.sender_email}&gt;\n"
            f"📅 <b>Data:</b> {e.date}\n"
            f"🎯 <b>Confiança:</b> {c.confidence:.1%}\n\n"
            f"📝 <b>Prévia:</b>\n{pe.preview}\n\n"
            f'🔗 <a href="{e.gmail_link}">Abrir no Gmail</a>'
        )
