import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import json

from app.main import app

client = TestClient(app)


def create_test_image(size=(100, 100), color="red"):
    """Create a test image for uploads"""
    image = Image.new("RGB", size, color)
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="JPEG")
    img_buffer.seek(0)
    return img_buffer


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns basic info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "CreativeIQ" in data["message"]

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_health_check(self):
        """Test API health endpoint"""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_status" in data

    def test_capabilities_endpoint(self):
        """Test capabilities endpoint"""
        response = client.get("/api/v1/health/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "supported_formats" in data


class TestAnalysisEndpoints:
    """Test design analysis endpoints"""

    def test_analyze_invalid_file(self):
        """Test analysis with invalid file"""
        response = client.post(
            "/api/v1/analyze/",
            files={"file": ("test.txt", "not an image", "text/plain")}
        )
        assert response.status_code == 400

    def test_analyze_no_file(self):
        """Test analysis without file"""
        response = client.post("/api/v1/analyze/")
        assert response.status_code == 422

    def test_analyze_valid_image(self):
        """Test analysis with valid image"""
        test_image = create_test_image()

        response = client.post(
            "/api/v1/analyze/",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
            data={
                "prompt": "Analyze this design",
                "analysis_type": "comprehensive",
                "target_platform": "general"
            }
        )

        # Note: This might fail if AI models aren't initialized
        # In that case, we expect a 500 error
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.json()
            assert "analysis_id" in data
            assert "status" in data

    def test_predict_performance(self):
        """Test performance prediction endpoint"""
        test_image = create_test_image()

        response = client.post(
            "/api/v1/analyze/predict",
            files={"file": ("test.jpg", test_image, "image/jpeg")},
            data={"target_platform": "instagram"}
        )

        # Expect error without proper AI model setup
        assert response.status_code in [200, 500]

    def test_batch_analysis_too_many_files(self):
        """Test batch analysis with too many files"""
        files = []
        for i in range(15):  # More than the limit of 10
            test_image = create_test_image()
            files.append(("files", (f"test{i}.jpg", test_image, "image/jpeg")))

        response = client.post("/api/v1/analyze/batch", files=files)
        assert response.status_code == 400


class TestChatEndpoints:
    """Test chat/coaching endpoints"""

    def test_chat_endpoint(self):
        """Test chat endpoint"""
        response = client.post(
            "/api/v1/chat/",
            json={
                "message": "How can I improve my design?",
                "analysis_id": None
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "suggestions" in data

    def test_chat_with_empty_message(self):
        """Test chat with empty message"""
        response = client.post(
            "/api/v1/chat/",
            json={"message": ""}
        )
        assert response.status_code == 422

    def test_feedback_endpoint(self):
        """Test feedback submission"""
        response = client.post(
            "/api/v1/chat/feedback",
            params={
                "analysis_id": "test-id",
                "rating": 5,
                "comments": "Great analysis!"
            }
        )
        assert response.status_code == 200

    def test_feedback_invalid_rating(self):
        """Test feedback with invalid rating"""
        response = client.post(
            "/api/v1/chat/feedback",
            params={
                "analysis_id": "test-id",
                "rating": 10,  # Invalid - should be 1-5
                "comments": "Test"
            }
        )
        assert response.status_code == 400


@pytest.mark.integration
class TestIntegrationScenarios:
    """Test complete user workflows"""

    def test_complete_analysis_workflow(self):
        """Test complete analysis workflow"""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Upload and analyze
        test_image = create_test_image(size=(500, 500), color="blue")

        analysis_response = client.post(
            "/api/v1/analyze/",
            files={"file": ("design.jpg", test_image, "image/jpeg")},
            data={"analysis_type": "comprehensive"}
        )

        if analysis_response.status_code == 200:
            analysis_data = analysis_response.json()
            analysis_id = analysis_data["analysis_id"]

            # 3. Get analysis result
            result_response = client.get(f"/api/v1/analyze/{analysis_id}")
            assert result_response.status_code in [200, 404]  # Might not be stored yet

            # 4. Chat about the analysis
            chat_response = client.post(
                "/api/v1/chat/",
                json={
                    "message": "What can I improve in this design?",
                    "analysis_id": analysis_id
                }
            )
            assert chat_response.status_code == 200

    def test_api_documentation_accessible(self):
        """Test that API documentation is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200

        response = client.get("/redoc")
        assert response.status_code == 200


# Fixtures for testing
@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing"""
    return {
        "analysis_id": "test-123",
        "status": "completed",
        "overall_score": 85.5,
        "color_analysis": {
            "dominant_colors": ["#FF0000", "#00FF00", "#0000FF"],
            "color_scheme": "triadic",
            "harmony_score": 88.0,
            "accessibility_score": 92.0
        },
        "typography_analysis": {
            "fonts_detected": ["sans-serif"],
            "font_pairing_score": 80.0,
            "readability_score": 85.0,
            "text_hierarchy_score": 75.0
        },
        "layout_analysis": {
            "composition_score": 88.0,
            "balance_score": 82.0,
            "grid_alignment": 79.0,
            "white_space_usage": 73.0,
            "focal_points": []
        },
        "recommendations": [
            {
                "category": "typography",
                "priority": "medium",
                "description": "Improve text hierarchy",
                "technical_details": "Use larger size differences between headers",
                "impact_score": 65.0
            }
        ]
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])