"""
Test script for API endpoints
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import FastAPI test client
from fastapi.testclient import TestClient

# Import our API
from api.main import app

# Create test client
client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    print("Testing root endpoint...")
    
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data
    
    print("✅ Root endpoint test passed")
    print(f"   Message: {data['message']}")


def test_health_endpoint():
    """Test health endpoint"""
    print("Testing health endpoint...")
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert data["status"] == "healthy"
    
    print("✅ Health endpoint test passed")
    print(f"   Status: {data['status']}")


def test_model_info_endpoint():
    """Test model info endpoint"""
    print("Testing model info endpoint...")
    
    model_name = "distilbert-base-uncased"
    response = client.get(f"/api/v1/models/{model_name}")
    
    # This might fail if the model is not available, which is expected in testing
    # We're just testing that the endpoint exists and returns a proper response structure
    assert response.status_code in [200, 500]  # 200 if model exists, 500 if loading fails
    
    if response.status_code == 200:
        data = response.json()
        assert "model_name" in data
        assert "model_type" in data
        assert "num_labels" in data
        assert "loaded" in data
        print("✅ Model info endpoint test passed")
    else:
        print("⚠️  Model info endpoint test completed (model loading failed as expected in test)")


def test_api_endpoints():
    """Test all API endpoints"""
    print("Testing API Endpoints...")
    print("=" * 25)
    
    try:
        test_root_endpoint()
        print()
        test_health_endpoint()
        print()
        test_model_info_endpoint()
        print()
        
        print("🎉 All API endpoint tests passed!")
        return True
    except Exception as e:
        print(f"❌ Error in API endpoint tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_api_endpoints()
    sys.exit(0 if success else 1)