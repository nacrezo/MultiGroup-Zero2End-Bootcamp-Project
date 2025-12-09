"""
Data loading and preprocessing utilities for Gaming User Segmentation.
This script helps download and prepare gaming user behavior datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR, TRAIN_FILE, TEST_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gaming_dataset(dataset_path: Path = None) -> pd.DataFrame:
    """
    Load and preprocess the real gaming user behavior dataset.
    Standardizes column names to match the application schema.
    """
    if dataset_path is None:
        dataset_path = RAW_DATA_DIR
    
    # Try to find the dataset file
    # Priority: Kaggle dataset first, then train.csv
    possible_files = [
        'online_gaming_behavior_dataset.csv',  # Kaggle dataset
        'predict_online_gaming_behavior_dataset.csv',
        'gaming_behavior.csv',
        'train.csv',  # Fallback to cleaned version
        'data.csv'
    ]
    train_file = None
    
    # Check for specific files first
    for file in possible_files:
        file_path = dataset_path / file
        if file_path.exists():
            train_file = file_path
            break
            
    # Fallback: Check any CSV
    if train_file is None:
        available_files = list(dataset_path.glob('*.csv'))
        if available_files:
            train_file = available_files[0]
        else:
            raise FileNotFoundError(f"No CSV file found in {dataset_path}. Please download the dataset.")
            
    logger.info(f"Loading dataset from {train_file}")
    df = pd.read_csv(train_file)
    
    # Standardize column names
    # The app expects: 'AchievementUnlocked', Kaggle has 'AchievementsUnlocked'
    rename_map = {
        'AchievementsUnlocked': 'AchievementUnlocked',  # Kaggle uses plural
        'achievements_unlocked': 'AchievementUnlocked',
        'user_id': 'PlayerID',
        'age': 'Age',
        'gender': 'Gender',
        'location': 'Location',
        'game_genre': 'GameGenre',
        'play_time_hours': 'PlayTimeHours',
        'in_game_purchases': 'InGamePurchases',
        'game_difficulty': 'GameDifficulty',
        'sessions_per_week': 'SessionsPerWeek',
        'avg_session_duration_minutes': 'AvgSessionDurationMinutes',
        'player_level': 'PlayerLevel',
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Drop EngagementLevel if it exists (not used in model, might be target variable)
    if 'EngagementLevel' in df.columns:
        logger.info("Dropping 'EngagementLevel' column (not used in segmentation model)")
        df = df.drop(columns=['EngagementLevel'])
    
    # Verify required columns exist
    required_cols = [
        'PlayerID', 'Age', 'Gender', 'Location', 'GameGenre', 
        'PlayTimeHours', 'InGamePurchases', 'GameDifficulty', 
        'SessionsPerWeek', 'AvgSessionDurationMinutes', 
        'PlayerLevel', 'AchievementUnlocked'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in dataset: {missing_cols}")
        # Optionally raise error if strict
        # raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    import sys
    import subprocess
    
    try:
        df = load_gaming_dataset(RAW_DATA_DIR)
        
        # Save standardized version to train.csv if it was loaded from another name
        if not TRAIN_FILE.exists() or pd.read_csv(TRAIN_FILE).shape != df.shape:
             try:
                 df.to_csv(TRAIN_FILE, index=False)
                 logger.info(f"Standardized dataset saved to {TRAIN_FILE}")
             except PermissionError:
                 logger.warning(f"Could not save to {TRAIN_FILE} (permission denied)")
                 
    except FileNotFoundError:
        logger.error("")
        logger.error("=" * 60)
        logger.error("Dataset bulunamadı!")
        logger.error("=" * 60)
        logger.error("")
        logger.error("Dataset dosyası data/raw/ klasöründe bulunamadı.")
        logger.error("")
        logger.error("ÇÖZÜM 1: Repo'yu yeniden clone edin (dataset repo'da olmalı)")
        logger.error("")
        logger.error("ÇÖZÜM 2: Dataset'i manuel olarak indirin:")
        logger.error("  1. https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
        logger.error("  2. 'Download' butonuna tıklayın")
        logger.error(f"  3. CSV dosyasını {RAW_DATA_DIR} klasörüne kopyalayın")
        logger.error("")
        logger.error("   Kabul edilen dosya isimleri:")
        logger.error("   - online_gaming_behavior_dataset.csv")
        logger.error("   - predict_online_gaming_behavior_dataset.csv")
        logger.error("   - gaming_behavior.csv")
        logger.error("   - train.csv")
        logger.error("   - data.csv")
        logger.error("")
        logger.error("=" * 60)
        sys.exit(1)


