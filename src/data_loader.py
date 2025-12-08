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


def download_kaggle_dataset(dataset_name: str, save_path: Path, files: list = None) -> None:
    """
    Download dataset from Kaggle.
    Requires kaggle API credentials in ~/.kaggle/kaggle.json
    
    Args:
        dataset_name: Kaggle dataset name (e.g., 'datasnaek/mobile-games-ab-testing')
        save_path: Directory to save the dataset
        files: Optional list of specific files to download
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        logger.info(f"Downloading dataset: {dataset_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        
        if files:
            for file in files:
                api.dataset_download_file(dataset_name, file, path=str(save_path))
        else:
            api.dataset_download_files(dataset_name, path=str(save_path), unzip=True)
        
        logger.info(f"Dataset downloaded successfully to {save_path}")
        
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        logger.info("Alternatively, you can manually download the dataset from Kaggle")
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        logger.info("Make sure you have:")
        logger.info("1. Kaggle API credentials in ~/.kaggle/kaggle.json")
        logger.info("2. kaggle package installed: pip install kaggle")
        raise


def load_gaming_dataset(dataset_path: Path = None) -> pd.DataFrame:
    """
    Load and preprocess the real gaming user behavior dataset.
    Standardizes column names to match the application schema.
    """
    if dataset_path is None:
        dataset_path = RAW_DATA_DIR
    
    # Try to find the dataset file
    possible_files = [
        'train.csv', 
        'data.csv', 
        'gaming_behavior.csv',
        'predict_online_gaming_behavior_dataset.csv'
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
    # The app expects: 'AchievementUnlocked', Kaggle might have 'AchievementsUnlocked'
    rename_map = {
        'AchievementsUnlocked': 'AchievementUnlocked',
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
        logger.error("Dataset not found. Please download it from Kaggle.")
        logger.info("Link: https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")


