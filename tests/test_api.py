import pytest

def test_health_check(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_root(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint(test_client, sample_user_data):
    response = test_client.post("/predict", json=sample_user_data)
    
    # If model is not loaded (e.g. CI env), it might return 503
    if response.status_code == 503:
        pytest.skip("Model not loaded in API")
        
    assert response.status_code == 200
    data = response.json()
    assert "cluster" in data
    assert "segment_name" in data
    assert isinstance(data["cluster"], int)

def test_predict_batch_endpoint(test_client, sample_user_data):
    batch_data = {"users": [sample_user_data, sample_user_data]}
    response = test_client.post("/predict/batch", json=batch_data)
    
    if response.status_code == 503:
        pytest.skip("Model not loaded in API")

    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert len(data["segments"]) == 2
