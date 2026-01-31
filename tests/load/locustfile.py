from locust import HttpUser, task, between
from io import BytesIO
from PIL import Image


class DroneDetectorUser(HttpUser):
    """Simulates users making requests to the drone detector API."""

    # Wait 1-3 seconds between requests
    wait_time = between(1, 3)

    def on_start(self):
        """Generate test images once when user starts (not per request)."""
        # Create a simple 224x224 RGB test image (standard size)
        img = Image.new("RGB", (224, 224), color=(73, 109, 137))
        img_bytes = BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        self.test_image = img_bytes.getvalue()

        # Create a larger image for edge case testing
        large_img = Image.new("RGB", (1920, 1080), color=(100, 150, 200))
        large_img_bytes = BytesIO()
        large_img.save(large_img_bytes, format="JPEG", quality=85)
        large_img_bytes.seek(0)
        self.large_test_image = large_img_bytes.getvalue()

    @task(1)
    def health_check(self):
        """Check API health - lightweight baseline."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(1)
    def root_endpoint(self):
        """Check root endpoint."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code in [200, 307, 404]:
                response.success()
            else:
                response.failure(f"Root endpoint failed: {response.status_code}")

    @task(2)
    def get_info(self):
        """Get API info."""
        with self.client.get("/v1/info", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "version" in data or "model" in data:
                        response.success()
                    else:
                        response.failure("Invalid info response format")
                except Exception as e:
                    response.failure(f"Info parsing failed: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(10)
    def predict_drone(self):
        """Make prediction with standard image - primary workload."""
        files = {"file": ("test.jpg", self.test_image, "image/jpeg")}
        with self.client.post("/v1/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "prediction" in data and "metadata" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Response parsing failed: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(3)
    def predict_large_image(self):
        """Make prediction with larger image - tests resizing."""
        files = {"file": ("large_test.jpg", self.large_test_image, "image/jpeg")}
        with self.client.post("/v1/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "prediction" in data:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except Exception as e:
                    response.failure(f"Response parsing failed: {e}")
            else:
                response.failure(f"Got status code {response.status_code}")

    @task(1)
    def openapi_docs(self):
        """Check OpenAPI documentation endpoint."""
        with self.client.get("/docs", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Docs endpoint failed: {response.status_code}")
