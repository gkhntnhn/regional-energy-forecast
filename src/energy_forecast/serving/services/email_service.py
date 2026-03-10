"""Email delivery service using SMTP."""

from __future__ import annotations

import smtplib
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from energy_forecast.serving.exceptions import EmailDeliveryError

if TYPE_CHECKING:
    from energy_forecast.monitoring.drift_detector import DriftAlert


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

Talep ettiğiniz 48 saatlik (T + T+1 gün) elektrik tüketimi tahmini ekte sunulmuştur.

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
        except OSError as e:
            logger.opt(exception=True).error(
                "SMTP socket error (likely Windows IPv6/IPv4 issue): {}", e
            )
            raise EmailDeliveryError(f"SMTP connection error: {e}") from e

    def send_with_retry(
        self,
        to_email: str,
        attachment_path: Path,
        job_id: str,
        created_at: str,
        max_retries: int = 2,
    ) -> tuple[bool, int, str | None]:
        """Send email with retry logic.

        Returns:
            (success, attempt_count, error_message_or_none)
        """
        if not self._enabled:
            return False, 0, "Email service disabled"

        last_error: str | None = None
        for attempt in range(1, max_retries + 1):
            try:
                msg = self._create_message(
                    to_email, attachment_path, job_id, created_at
                )
                self._send_message(msg, to_email)
                logger.info(
                    "Email sent to {} (attempt {})", to_email, attempt
                )
                return True, attempt, None
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Email attempt {}/{} failed: {}",
                    attempt,
                    max_retries,
                    e,
                )

        return False, max_retries, last_error

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

    def send_drift_alert(self, admin_email: str, alert: DriftAlert) -> bool:
        """Send model drift notification email.

        Args:
            admin_email: Admin recipient address.
            alert: Drift alert with type, severity, and message.

        Returns:
            True if sent successfully.
        """
        if not self._enabled:
            logger.warning("Drift alert not sent (service disabled)")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = (
                f"{self._config.sender_name} <{self._config.sender_email}>"
            )
            msg["To"] = admin_email
            msg["Subject"] = (
                f"[Energy Forecast] Drift Alert: "
                f"{alert.alert_type} ({alert.severity})"
            )

            action_map = {
                "mape_threshold": "Model retrain degerlendir",
                "mape_trend": "Feature/data kalitesi kontrol et",
                "bias_shift": "Bolge tuketim profili degismis olabilir",
            }
            action = action_map.get(alert.alert_type, "Durumu inceleyin")

            body = (
                f"Model Drift Tespit Edildi\n"
                f"========================\n\n"
                f"Tip: {alert.alert_type}\n"
                f"Ciddiyet: {alert.severity.upper()}\n"
                f"Mevcut deger: {alert.current_value:.2f}\n"
                f"Threshold: {alert.threshold:.2f}\n"
                f"Pencere: Son {alert.window_days} gun\n\n"
                f"Mesaj: {alert.message}\n\n"
                f"Onerilen aksiyon: {action}\n\n"
                f"Bu email otomatik uretilmistir."
            )

            msg.attach(MIMEText(body, "plain", "utf-8"))
            self._send_message(msg, admin_email)
            logger.info(
                "Drift alert sent to {}: {} ({})",
                admin_email,
                alert.alert_type,
                alert.severity,
            )
            return True

        except Exception as e:
            logger.error("Failed to send drift alert: {}", e)
            return False
