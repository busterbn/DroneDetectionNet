import logging
from drone_detector_mlops.utils.logger import (
    SUCCESS_LEVEL,
    AppLogger,
    get_logger,
)


def test_success_level_value():
    """Test that SUCCESS_LEVEL is correctly defined."""
    assert SUCCESS_LEVEL == 25
    assert logging.getLevelName(SUCCESS_LEVEL) == "SUCCESS"


def test_app_logger_initialization():
    """Test that AppLogger can be initialized."""
    logger = AppLogger(name="test_logger")
    assert logger.logger is not None
    assert logger.logger.name == "test_logger"


def test_app_logger_info_method(capsys):
    """Test that info method logs correctly."""
    logger = AppLogger(name="test_info")
    logger.info("Test info message")

    captured = capsys.readouterr()
    assert "Test info message" in captured.out
    assert "INFO" in captured.out


def test_app_logger_success_method(capsys):
    """Test that success method logs correctly."""
    logger = AppLogger(name="test_success")
    logger.success("Test success message")

    captured = capsys.readouterr()
    assert "Test success message" in captured.out
    assert "SUCCESS" in captured.out


def test_get_logger_returns_app_logger():
    """Test that get_logger returns an AppLogger instance."""
    logger = get_logger("test_module")
    assert isinstance(logger, AppLogger)


def test_app_logger_info_with_context(capsys):
    """Test logging info with context kwargs."""
    logger = AppLogger(name="test_context")
    logger.info("Message", user="alice", count=5)

    captured = capsys.readouterr()
    assert "Message" in captured.out
    assert "user=alice" in captured.out
    assert "count=5" in captured.out


def test_app_logger_warning_method(capsys):
    """Test that warning method logs correctly."""
    logger = AppLogger(name="test_warning")
    logger.warning("Warning message")

    captured = capsys.readouterr()
    assert "Warning message" in captured.out
    assert "WARNING" in captured.out


def test_app_logger_error_method(capsys):
    """Test that error method logs correctly."""
    logger = AppLogger(name="test_error")
    logger.error("Error message")

    captured = capsys.readouterr()
    assert "Error message" in captured.out
    assert "ERROR" in captured.out
