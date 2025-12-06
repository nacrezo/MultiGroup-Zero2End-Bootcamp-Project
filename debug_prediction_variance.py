
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.append(str(Path.cwd()))

from src.pipeline import UserSegmentationPipeline

def test_prediction_variance():
    pipeline = UserSegmentationPipeline()
    pipeline.load()
    
    print(f"Model loaded. Clusters: {pipeline.n_clusters}")
    print(f"Expected Features ({len(pipeline.feature_names)}): {pipeline.feature_names}")

    # Define diverse profiles
    profiles = [
        {
            "name": "Casual Action Player",
            "data": {
                'PlayerID': 1, 'Age': 20, 'Gender': 'Male', 'Location': 'USA',
                'GameGenre': 'Action', 'GameDifficulty': 'Easy',
                'PlayTimeHours': 2.0, 'InGamePurchases': 0, 'SessionsPerWeek': 1,
                'AvgSessionDurationMinutes': 20.0, 'PlayerLevel': 5, 'AchievementUnlocked': 2
            }
        },
        {
            "name": "Hardcore Strategy Whale",
            "data": {
                'PlayerID': 2, 'Age': 30, 'Gender': 'Female', 'Location': 'Europe',
                'GameGenre': 'Strategy', 'GameDifficulty': 'Hard',
                'PlayTimeHours': 50.0, 'InGamePurchases': 1, 'SessionsPerWeek': 20,
                'AvgSessionDurationMinutes': 120.0, 'PlayerLevel': 90, 'AchievementUnlocked': 100
            }
        },
        {
            "name": "Mid-Core RPG Player",
            "data": {
                'PlayerID': 3, 'Age': 25, 'Gender': 'Male', 'Location': 'Asia',
                'GameGenre': 'RPG', 'GameDifficulty': 'Medium',
                'PlayTimeHours': 15.0, 'InGamePurchases': 0, 'SessionsPerWeek': 5,
                'AvgSessionDurationMinutes': 60.0, 'PlayerLevel': 40, 'AchievementUnlocked': 30
            }
        },
        {
            "name": "Simulation Fan",
            "data": {
                'PlayerID': 4, 'Age': 40, 'Gender': 'Female', 'Location': 'Other',
                'GameGenre': 'Simulation', 'GameDifficulty': 'Easy',
                'PlayTimeHours': 10.0, 'InGamePurchases': 0, 'SessionsPerWeek': 3,
                'AvgSessionDurationMinutes': 45.0, 'PlayerLevel': 20, 'AchievementUnlocked': 10
            }
        }
    ]
    
    print("\n--- Testing Predictions ---")
    for p in profiles:
        df = pd.DataFrame([p['data']])
        
        # Checking preprocessing first
        processed = pipeline.preprocess(df, is_training=False)
        print(f"\n{p['name']} Pre-alignment cols: {[c for c in processed.columns if 'GameGenre' in c]}")
        
        # Create full feature vector like predict does
        X = processed.reindex(columns=pipeline.feature_names, fill_value=0)
        
        # Get raw prediction
        scaled = pipeline.scaler.transform(X)
        segment = pipeline.model.predict(scaled)[0]
        
        print(f"\n{p['name']} -> Predicted Segment: {segment}")
        print(f"  Genre Input: {p['data']['GameGenre']}")
        
        # Check specific OHE columns
        genre_cols = [c for c in X.columns if 'GameGenre' in c]
        print(f"  Genre Cols: {genre_cols}")
        print(f"  Values: {[X.iloc[0][c] for c in genre_cols]}")

if __name__ == "__main__":
    test_prediction_variance()
