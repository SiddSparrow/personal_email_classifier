import base64
import logging
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.interfaces import EmailReader, EmailMessage

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
]


class GmailReader(EmailReader):
    """Reads emails from Gmail via the official API with OAuth2."""

    def __init__(
        self,
        credentials_file: Path,
        token_file: Path,
        query: str,
    ) -> None:
        self._credentials_file = credentials_file
        self._token_file = token_file
        self._query = query
        self._service = None

    def _get_service(self):
        """Authenticate and build Gmail service (lazy, cached)."""
        if self._service is not None:
            return self._service

        creds = None
        if self._token_file.exists():
            creds = Credentials.from_authorized_user_file(
                str(self._token_file), SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._credentials_file), SCOPES
                )
                creds = flow.run_local_server(port=0)

            self._token_file.parent.mkdir(parents=True, exist_ok=True)
            self._token_file.write_text(creds.to_json())

        self._service = build("gmail", "v1", credentials=creds)
        return self._service

    def fetch_unread(self, max_results: int) -> list[EmailMessage]:
        service = self._get_service()
        results = (
            service.users()
            .messages()
            .list(userId="me", q=self._query, maxResults=max_results)
            .execute()
        )
        messages = results.get("messages", [])
        logger.info(f"Found {len(messages)} unread message(s)")

        emails = []
        for msg_stub in messages:
            try:
                email = self._parse_message(service, msg_stub["id"])
                emails.append(email)
            except Exception:
                logger.exception(f"Failed to parse message {msg_stub['id']}")
        return emails

    def mark_as_read(self, message_id: str) -> None:
        service = self._get_service()
        service.users().messages().modify(
            userId="me",
            id=message_id,
            body={"removeLabelIds": ["UNREAD"]},
        ).execute()
        logger.debug(f"Marked {message_id} as read")

    def _parse_message(self, service, message_id: str) -> EmailMessage:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}

        subject = headers.get("subject", "(no subject)")
        from_raw = headers.get("from", "")
        date = headers.get("date", "")

        sender_name, sender_email = self._parse_from(from_raw)
        body_html, body_plain = self._extract_body(msg["payload"])
        gmail_link = f"https://mail.google.com/mail/u/0/#inbox/{message_id}"

        return EmailMessage(
            message_id=message_id,
            subject=subject,
            sender_name=sender_name,
            sender_email=sender_email,
            date=date,
            body_html=body_html,
            body_plain=body_plain,
            gmail_link=gmail_link,
        )

    @staticmethod
    def _parse_from(from_raw: str) -> tuple[str, str]:
        """Parse 'Display Name <email@example.com>' into (name, email)."""
        if "<" in from_raw and ">" in from_raw:
            name = from_raw[: from_raw.index("<")].strip().strip('"')
            email = from_raw[from_raw.index("<") + 1 : from_raw.index(">")]
            return name, email
        return "", from_raw.strip()

    @staticmethod
    def _extract_body(payload: dict) -> tuple[str, str]:
        """Recursively extract HTML and plain text body from MIME payload."""
        body_html = ""
        body_plain = ""

        def _walk(part):
            nonlocal body_html, body_plain
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data", "")

            if data:
                decoded = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if mime == "text/html" and not body_html:
                    body_html = decoded
                elif mime == "text/plain" and not body_plain:
                    body_plain = decoded

            for sub in part.get("parts", []):
                _walk(sub)

        _walk(payload)
        return body_html, body_plain
