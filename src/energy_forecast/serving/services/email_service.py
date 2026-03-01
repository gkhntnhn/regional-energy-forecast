"""Email delivery service using SMTP."""

from __future__ import annotations

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from energy_forecast.serving.exceptions import EmailDeliveryError


class EmailServiceConfig(BaseModel, frozen=True):
    """Email service configuration."""

    smtp_server: str = Field(default="")
    smtp_port: int = Field(default=587)
    username: str = Field(default="")
    password: str = Field(default="")
    sender_email: EmailStr | str = Field(default="noreply@example.com")
    sender_name: str = Field(default="Energy Forecast")
    use_tls: bool = Field(default=True)
    timeout: int = Field(default=30)

    # Email templates
    subject_template: str = Field(default="Tahmin Sonuçları - {job_id}")
    body_template: str = Field(
        default="""Merhaba,

Talep ettiğiniz 24 saatlik (T+1 gün) elektrik tüketimi tahmini ekte sunulmuştur.

İş No: {job_id}
Oluşturulma: {created_at}

İyi çalışmalar,
Energy Forecast Sistemi"""
    )


class EmailService:
    """Sends prediction results via SMTP email.

    Uses standard smtplib for synchronous sending. This is safe
    because it runs in BackgroundTasks which uses a thread pool.

    Args:
        config: Email service configuration.
    """

    def __init__(self, config: EmailServiceConfig) -> None:
        self._config = config
        self._enabled = bool(config.smtp_server and config.username)

        if not self._enabled:
            logger.warning(
                "Email service disabled: SMTP server or username not configured"
            )

    @property
    def is_enabled(self) -> bool:
        """Check if email service is properly configured."""
        return self._enabled

    def send_prediction_result(
        self,
        to_email: str,
        attachment_path: Path,
        job_id: str,
        created_at: str,
    ) -> bool:
        """Send prediction results email with attachment.

        Args:
            to_email: Recipient email address.
            attachment_path: Path to Excel file to attach.
            job_id: Job identifier for subject/body.
            created_at: Job creation timestamp string.

        Returns:
            True if sent successfully.

        Raises:
            EmailDeliveryError: If sending fails.
        """
        if not self._enabled:
            logger.warning("Email not sent (service disabled): {}", to_email)
            return False

        try:
            msg = self._create_message(to_email, attachment_path, job_id, created_at)
            self._send_message(msg, to_email)
            logger.info("Email sent successfully to {}", to_email)
            return True

        except EmailDeliveryError:
            raise
        except Exception as e:
            logger.error("Failed to send email to {}: {}", to_email, e)
            raise EmailDeliveryError(f"Failed to send email: {e}") from e

    def _create_message(
        self,
        to_email: str,
        attachment_path: Path,
        job_id: str,
        created_at: str,
    ) -> MIMEMultipart:
        """Create email message with attachment."""
        msg = MIMEMultipart()

        # Headers
        msg["From"] = f"{self._config.sender_name} <{self._config.sender_email}>"
        msg["To"] = to_email
        msg["Subject"] = self._config.subject_template.format(job_id=job_id)

        # Body
        body = self._config.body_template.format(
            job_id=job_id,
            created_at=created_at,
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # Attachment
        if attachment_path.exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={attachment_path.name}",
                )
                msg.attach(part)

        return msg

    def _send_message(self, msg: MIMEMultipart, to_email: str) -> None:
        """Send message via SMTP."""
        try:
            with smtplib.SMTP(
                self._config.smtp_server,
                self._config.smtp_port,
                timeout=self._config.timeout,
            ) as server:
                if self._config.use_tls:
                    server.starttls()

                server.login(self._config.username, self._config.password)
                server.sendmail(
                    str(self._config.sender_email),
                    to_email,
                    msg.as_string(),
                )

        except smtplib.SMTPAuthenticationError as e:
            raise EmailDeliveryError("SMTP authentication failed") from e
        except smtplib.SMTPRecipientsRefused as e:
            raise EmailDeliveryError(f"Recipient refused: {to_email}") from e
        except smtplib.SMTPException as e:
            raise EmailDeliveryError(f"SMTP error: {e}") from e
        except TimeoutError as e:
            raise EmailDeliveryError("SMTP connection timeout") from e

    def send_error_notification(
        self,
        to_email: str,
        job_id: str,
        error_message: str,
    ) -> bool:
        """Send error notification email.

        Args:
            to_email: Recipient email address.
            job_id: Failed job identifier.
            error_message: Error description.

        Returns:
            True if sent successfully.
        """
        if not self._enabled:
            logger.warning("Error email not sent (service disabled): {}", to_email)
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = f"{self._config.sender_name} <{self._config.sender_email}>"
            msg["To"] = to_email
            msg["Subject"] = f"Tahmin Hatası - {job_id}"

            body = f"""Merhaba,

Talep ettiğiniz tahmin işlemi sırasında bir hata oluştu.

İş No: {job_id}
Hata: {error_message}

Lütfen daha sonra tekrar deneyin veya yönetici ile iletişime geçin.

Energy Forecast Sistemi"""

            msg.attach(MIMEText(body, "plain", "utf-8"))
            self._send_message(msg, to_email)
            logger.info("Error notification sent to {}", to_email)
            return True

        except Exception as e:
            logger.error("Failed to send error notification: {}", e)
            return False
