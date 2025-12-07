import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.app import app
from src.pipeline import UserSegmentationPipeline

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def sample_user_data():
    return {
        'PlayerID': 1001,
        'Age': 25,
        'Gender': 'Male',
        'Location': 'USA',
        'GameGenre': 'Action',
        'PlayTimeHours': 35.5,
        'InGamePurchases': 1,
        'GameDifficulty': 'Medium',
        'SessionsPerWeek': 5,
        'AvgSessionDurationMinutes': 45.0,
        'PlayerLevel': 10,
        'AchievementUnlocked': 15
    }

@pytest.fixture
def sample_df(sample_user_data):
    return pd.DataFrame([sample_user_data])
