"""
Inference script for Gaming User Segmentation.
Predicts user segments for new data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, Any, Optional

sys.path.append(str(Path(__file__).parent.parent))
from src.config import *
from src.pipeline import UserSegmentationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_user_segment(user_data: Dict[str, Any], pipeline: Optional[UserSegmentationPipeline] = None) -> int:
    """
    Predict user segment for a single user.
    
    Args:
        user_data: Dictionary containing user features
        pipeline: Trained pipeline (if None, loads from disk)
    
    Returns:
        Cluster label (segment ID)
    """
    if pipeline is None:
        pipeline = UserSegmentationPipeline()
        pipeline.load()
    
    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    
    # Predict
    cluster = pipeline.predict(df)[0]
    
    return int(cluster)


def predict_batch(df: pd.DataFrame, pipeline: Optional[UserSegmentationPipeline] = None) -> pd.Series:
    """
    Predict user segments for a batch of users.
    
    Args:
        df: DataFrame containing user features
        pipeline: Trained pipeline (if None, loads from disk)
    
    Returns:
        Series of cluster labels
    """
    if pipeline is None:
        pipeline = UserSegmentationPipeline()
        pipeline.load()
    
    # Predict
    clusters = pipeline.predict(df)
    
    return pd.Series(clusters, index=df.index, name='cluster')


def get_segment_profile(cluster_id: int, df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Get profile statistics for a segment.
    
    Args:
        cluster_id: Cluster/segment ID
        df: DataFrame with cluster labels (if None, loads from disk)
    
    Returns:
        Dictionary with segment statistics
    """
    if df is None:
        processed_path = PROCESSED_DATA_DIR / "segmented_users.csv"
        if processed_path.exists():
            df = pd.read_csv(processed_path)
        else:
            raise FileNotFoundError("Processed data not found. Please train the model first.")
    
    cluster_data = df[df['cluster'] == cluster_id]
    
    if len(cluster_data) == 0:
        return {"error": f"Cluster {cluster_id} not found"}
    
    profile = {
        'cluster_id': cluster_id,
        'user_count': len(cluster_data),
        'percentage': len(cluster_data) / len(df) * 100,
    }
    
    # Add key metrics
    key_features = ['total_sessions', 'total_playtime_hours', 'total_spent_usd', 
                    'login_frequency_per_week', 'engagement_score']
    
    for feature in key_features:
        if feature in cluster_data.columns:
            profile[f'avg_{feature}'] = float(cluster_data[feature].mean())
            profile[f'median_{feature}'] = float(cluster_data[feature].median())
    
    return profile


if __name__ == "__main__":
    # Example usage
    logger.info("Loading pipeline...")
    pipeline = UserSegmentationPipeline()
    pipeline.load()
    
    # Example: Predict for a single user
    example_user = {
        'age': 25,
        'gender': 'Male',
        'country': 'USA',
        'device_type': 'Mobile',
        'total_sessions': 50,
        'total_playtime_hours': 120,
        'total_spent_usd': 50,
        'login_frequency_per_week': 5,
        'max_level_reached': 30,
        'levels_completed': 25,
        'quests_completed': 40,
        'achievements_unlocked': 10,
        'days_since_last_login': 2,
        'days_since_registration': 60,
        'friend_count': 5,
        'guild_member': 1,
        'purchase_count': 3,
        'premium_subscription': 0,
        'win_rate': 0.6,
        'avg_score': 1500,
    }
    
    cluster = predict_user_segment(example_user, pipeline)
    print(f"\nPredicted segment for example user: {cluster}")
    
    # Get segment profile
    profile = get_segment_profile(cluster)
    print(f"\nSegment {cluster} Profile:")
    for key, value in profile.items():
        print(f"  {key}: {value}")

