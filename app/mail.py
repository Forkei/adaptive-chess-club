"""Minimal outbound-mail helper (Phase 4.0a).

If SMTP settings are filled in `.env` we send via SMTP. Otherwise we log
the message to the app logger AND append it to `<log_dir>/outbox.log` so
a developer can pick reset links off disk without needing real mail
during early testing.
"""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Email:
    to: str
    subject: str
    body: str


def _smtp_configured() -> bool:
    s = get_settings()
    return bool(s.smtp_host and s.smtp_from)


def _write_dev_outbox(email: Email) -> None:
    try:
        outbox = get_settings().log_path / "outbox.log"
        with outbox.open("a", encoding="utf-8") as fh:
            fh.write(
                "--- outbound email ---\n"
                f"to: {email.to}\n"
                f"subject: {email.subject}\n"
                f"body:\n{email.body}\n\n"
            )
    except Exception:
        logger.exception("Failed writing dev outbox")


def send(email: Email) -> None:
    """Fire-and-forget send. Never raises on failure — logs and moves on.

    Call sites should not depend on delivery success (no-one is waiting
    on an HTTP response that promises "email sent").
    """
    if not _smtp_configured():
        logger.info("DEV mail (not sent): to=%s subject=%s", email.to, email.subject)
        logger.info("DEV mail body:\n%s", email.body)
        _write_dev_outbox(email)
        return

    s = get_settings()
    msg = EmailMessage()
    msg["From"] = s.smtp_from
    msg["To"] = email.to
    msg["Subject"] = email.subject
    msg.set_content(email.body)

    try:
        with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=10) as server:
            server.starttls()
            if s.smtp_user:
                server.login(s.smtp_user, s.smtp_password)
            server.send_message(msg)
    except Exception:
        logger.exception("SMTP send failed for to=%s subject=%s", email.to, email.subject)
