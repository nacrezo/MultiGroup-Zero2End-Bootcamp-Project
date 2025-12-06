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
    Load and preprocess a gaming user behavior dataset.
    This function handles common gaming dataset formats.
    """
    if dataset_path is None:
        dataset_path = RAW_DATA_DIR
    
    # Try to find the dataset file
    possible_files = [
        'train.csv', 
        'data.csv', 
        'gaming_data.csv', 
        'user_data.csv', 
        'player_data.csv',
        'game_data.csv',
        'ab_test.csv',
        'cookie_cats.csv'
    ]
    train_file = None
    
    for file in possible_files:
        file_path = dataset_path / file
        if file_path.exists():
            train_file = file_path
            break
    
    if train_file is None:
        # List available files
        available_files = list(dataset_path.glob('*.csv'))
        if available_files:
            train_file = available_files[0]
            logger.info(f"Using file: {train_file.name}")
        else:
            raise FileNotFoundError(f"No CSV file found in {dataset_path}")
    
    logger.info(f"Loading dataset from {train_file}")
    df = pd.read_csv(train_file)
    
    logger.info(f"Dataset loaded. Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    return df


def create_sample_gaming_dataset(n_samples: int = 20000, save_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create a sample gaming user behavior dataset for demonstration.
    This is a synthetic dataset that mimics real gaming user data.
    
    Features include:
    - User demographics
    - Gameplay behavior (sessions, playtime, levels)
    - In-game purchases
    - Engagement metrics
    - Retention metrics
    """
    np.random.seed(42)
    
    n_samples = n_samples
    
    # Generate user features
    data = {
        'user_id': [f'user_{i:06d}' for i in range(1, n_samples + 1)],
        'age': np.random.normal(28, 8, n_samples).astype(int),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.55, 0.40, 0.05]),
        'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Turkey', 'Japan', 'South Korea'], 
                                    n_samples, p=[0.25, 0.15, 0.12, 0.10, 0.10, 0.15, 0.13]),
        'device_type': np.random.choice(['Mobile', 'PC', 'Console'], n_samples, p=[0.60, 0.25, 0.15]),
        'registration_date': pd.date_range('2023-01-01', periods=n_samples, freq='h')[:n_samples],
        
        # Gameplay behavior
        'total_sessions': np.random.poisson(45, n_samples),
        'total_playtime_hours': np.random.lognormal(3.5, 1.2, n_samples),
        'avg_session_duration_minutes': np.random.lognormal(2.5, 0.8, n_samples),
        'max_level_reached': np.random.poisson(25, n_samples),
        'levels_completed': np.random.poisson(20, n_samples),
        'quests_completed': np.random.poisson(35, n_samples),
        'achievements_unlocked': np.random.poisson(12, n_samples),
        
        # Engagement metrics
        'days_since_last_login': np.random.exponential(5, n_samples).astype(int),
        'days_since_registration': np.random.exponential(60, n_samples).astype(int),
        'login_frequency_per_week': np.random.lognormal(1.5, 0.9, n_samples),
        'friend_count': np.random.poisson(8, n_samples),
        'guild_member': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        
        # In-game purchases
        'total_spent_usd': np.random.lognormal(2, 2, n_samples),
        'purchase_count': np.random.poisson(3, n_samples),
        'avg_purchase_value': np.random.lognormal(1.5, 1, n_samples),
        'last_purchase_days_ago': np.random.exponential(20, n_samples).astype(int),
        'premium_subscription': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        
        # Game performance
        'win_rate': np.random.beta(3, 2, n_samples),
        'avg_score': np.random.normal(1500, 400, n_samples),
        'pvp_matches_played': np.random.poisson(15, n_samples),
        'pve_missions_completed': np.random.poisson(40, n_samples),
        
        # Social features
        'chat_messages_sent': np.random.poisson(50, n_samples),
        'reviews_written': np.random.poisson(2, n_samples),
        'events_participated': np.random.poisson(5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Ensure realistic values
    df['age'] = df['age'].clip(13, 65)
    df['total_playtime_hours'] = df['total_playtime_hours'].clip(0.1, 1000)
    df['avg_session_duration_minutes'] = df['avg_session_duration_minutes'].clip(1, 300)
    df['max_level_reached'] = df['max_level_reached'].clip(1, 100)
    df['total_spent_usd'] = df['total_spent_usd'].clip(0, 5000)
    df['login_frequency_per_week'] = df['login_frequency_per_week'].clip(0.1, 50)
    
    # Add some missing values (realistic scenario)
    missing_cols = ['friend_count', 'chat_messages_sent', 'reviews_written']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(len(df) * 0.03), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Create derived features that might be useful
    df['playtime_per_session'] = df['total_playtime_hours'] / (df['total_sessions'] + 1)
    df['spending_per_session'] = df['total_spent_usd'] / (df['total_sessions'] + 1)
    df['level_completion_rate'] = df['levels_completed'] / (df['max_level_reached'] + 1)
    df['engagement_score'] = (
        df['total_sessions'] * 0.3 +
        df['total_playtime_hours'] * 0.2 +
        df['login_frequency_per_week'] * 0.2 +
        df['quests_completed'] * 0.15 +
        df['achievements_unlocked'] * 0.15
    )
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Sample gaming dataset saved to {save_path}")
    
    logger.info(f"Dataset created with shape: {df.shape}")
    logger.info(f"Features: {df.columns.tolist()}")
    
    return df


if __name__ == "__main__":
    # Try to load real dataset first, then create sample if needed
    try:
        if TRAIN_FILE.exists():
            logger.info(f"Loading existing dataset from {TRAIN_FILE}")
            df = pd.read_csv(TRAIN_FILE)
            logger.info(f"Dataset loaded. Shape: {df.shape}")
        else:
            # Try to load from raw data directory
            df = load_gaming_dataset(RAW_DATA_DIR)
            if df is not None:
                df.to_csv(TRAIN_FILE, index=False)
                logger.info(f"Dataset saved to {TRAIN_FILE}")
    except FileNotFoundError:
        logger.info("No existing dataset found. Creating sample gaming dataset...")
        df = create_sample_gaming_dataset(n_samples=20000, save_path=TRAIN_FILE)
        logger.info(f"Sample dataset created with shape: {df.shape}")

