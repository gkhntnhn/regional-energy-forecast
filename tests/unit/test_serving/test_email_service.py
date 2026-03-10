"""Tests for email service."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from energy_forecast.serving.exceptions import EmailDeliveryError
from energy_forecast.serving.services.email_service import EmailService, EmailServiceConfig


@pytest.fixture
def email_config() -> EmailServiceConfig:
    """Create email config for testing."""
    return EmailServiceConfig(
        smtp_server="smtp.test.com",
        smtp_port=587,
        username="test@test.com",
        password="testpass",
        sender_email="noreply@test.com",
        sender_name="Test Forecast",
    )


@pytest.fixture
def disabled_email_config() -> EmailServiceConfig:
    """Create disabled email config."""
    return EmailServiceConfig()  # No SMTP server = disabled


@pytest.fixture
def email_service(email_config: EmailServiceConfig) -> EmailService:
    """Create email service for testing."""
    return EmailService(email_config)


@pytest.fixture
def sample_attachment(tmp_path: Path) -> Path:
    """Create sample attachment file."""
    attachment = tmp_path / "forecast.xlsx"
    attachment.write_bytes(b"test excel content")
    return attachment


class TestEmailServiceConfig:
    """Tests for EmailServiceConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = EmailServiceConfig()
        assert config.smtp_server == ""
        assert config.smtp_port == 587
        assert config.use_tls is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EmailServiceConfig(
            smtp_server="mail.example.com",
            smtp_port=465,
            use_tls=False,
        )
        assert config.smtp_server == "mail.example.com"
        assert config.smtp_port == 465


class TestEmailService:
    """Tests for EmailService."""

    def test_is_enabled(self, email_service: EmailService) -> None:
        """Test service is enabled with valid config."""
        assert email_service.is_enabled is True

    def test_is_disabled(self, disabled_email_config: EmailServiceConfig) -> None:
        """Test service is disabled without SMTP config."""
        service = EmailService(disabled_email_config)
        assert service.is_enabled is False

    def test_send_disabled_returns_false(
        self,
        disabled_email_config: EmailServiceConfig,
        sample_attachment: Path,
    ) -> None:
        """Test sending when disabled returns False."""
        service = EmailService(disabled_email_config)
        result = service.send_prediction_result(
            to_email="user@test.com",
            attachment_path=sample_attachment,
            job_id="test123",
            created_at="2025-01-01 10:00:00",
        )
        assert result is False

    @patch("energy_forecast.serving.services.email_service.smtplib.SMTP")
    def test_send_prediction_result_success(
        self,
        mock_smtp: MagicMock,
        email_service: EmailService,
        sample_attachment: Path,
    ) -> None:
        """Test successful email sending."""
        # Configure mock
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        result = email_service.send_prediction_result(
            to_email="user@test.com",
            attachment_path=sample_attachment,
            job_id="test123",
            created_at="2025-01-01 10:00:00",
        )

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch("energy_forecast.serving.services.email_service.smtplib.SMTP")
    def test_send_prediction_result_smtp_error(
        self,
        mock_smtp: MagicMock,
        email_service: EmailService,
        sample_attachment: Path,
    ) -> None:
        """Test email sending with SMTP error."""
        import smtplib

        mock_smtp.return_value.__enter__ = MagicMock(
            side_effect=smtplib.SMTPException("Connection failed")
        )

        with pytest.raises(EmailDeliveryError):
            email_service.send_prediction_result(
                to_email="user@test.com",
                attachment_path=sample_attachment,
                job_id="test123",
                created_at="2025-01-01 10:00:00",
            )

    @patch("energy_forecast.serving.services.email_service.smtplib.SMTP")
    def test_send_prediction_result_os_error(
        self,
        mock_smtp: MagicMock,
        email_service: EmailService,
        sample_attachment: Path,
    ) -> None:
        """Test OSError (Errno 22) during SMTP connection raises EmailDeliveryError."""
        mock_smtp.side_effect = OSError(22, "Invalid argument")

        with pytest.raises(EmailDeliveryError, match="SMTP connection error"):
            email_service.send_prediction_result(
                to_email="user@test.com",
                attachment_path=sample_attachment,
                job_id="test123",
                created_at="2025-01-01 10:00:00",
            )

    @patch("energy_forecast.serving.services.email_service.smtplib.SMTP")
    def test_send_with_retry_returns_failure_on_os_error(
        self,
        mock_smtp: MagicMock,
        email_service: EmailService,
        sample_attachment: Path,
    ) -> None:
        """Test send_with_retry gracefully handles OSError across retries."""
        mock_smtp.side_effect = OSError(22, "Invalid argument")

        success, attempts, error_msg = email_service.send_with_retry(
            to_email="user@test.com",
            attachment_path=sample_attachment,
            job_id="test123",
            created_at="2025-01-01 10:00:00",
            max_retries=2,
        )

        assert success is False
        assert attempts == 2
        assert error_msg is not None
        assert "SMTP connection error" in error_msg

    @patch("energy_forecast.serving.services.email_service.smtplib.SMTP")
    def test_send_error_notification(
        self,
        mock_smtp: MagicMock,
        email_service: EmailService,
    ) -> None:
        """Test error notification email."""
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

        result = email_service.send_error_notification(
            to_email="user@test.com",
            job_id="test123",
            error_message="Something went wrong",
        )

        assert result is True
        mock_server.sendmail.assert_called_once()
