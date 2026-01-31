from drone_detector_mlops.api.schemas import HealthResponse, InfoResponse


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_status(self):
        """Test healthy status response."""
        response = HealthResponse(status="healthy", model_loaded=True)
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_unhealthy_status(self):
        """Test unhealthy status response."""
        response = HealthResponse(status="unhealthy", model_loaded=False)
        assert response.status == "unhealthy"
        assert response.model_loaded is False


class TestInfoResponse:
    """Tests for InfoResponse schema."""

    def test_valid_info_response(self):
        """Test creating valid info response."""
        response = InfoResponse(model_version="model_v1.onnx", uptime_seconds=123.45)
        assert response.model_version == "model_v1.onnx"
        assert response.uptime_seconds == 123.45

    def test_uptime_as_float(self):
        """Test that uptime is properly converted to float."""
        response = InfoResponse(model_version="model_v1.onnx", uptime_seconds=100)
        assert isinstance(response.uptime_seconds, float)
