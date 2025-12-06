"""
Kaggle dataset indirme scripti
Gaming user segmentation için uygun dataset indirir
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import RAW_DATA_DIR
from src.data_loader import download_kaggle_dataset, create_sample_gaming_dataset, load_gaming_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gaming user segmentation için önerilen dataset'ler
GAMING_DATASETS = {
    'cookie-cats': {
        'name': 'datasnaek/mobile-games-ab-testing',
        'description': 'Mobile Games A/B Testing - Cookie Cats (Popular gaming dataset)'
    },
    'player-behavior': {
        'name': 'aravindh1/player-behavior-dataset',
        'description': 'Player Behavior Dataset (if available)'
    },
    # Kullanıcı kendi dataset'ini ekleyebilir
}

def main():
    """Download or create gaming dataset."""
    logger.info("=" * 60)
    logger.info("Gaming User Segmentation - Dataset Downloader")
    logger.info("=" * 60)
    
    # Önce mevcut dataset'i kontrol et
    try:
        df = load_gaming_dataset(RAW_DATA_DIR)
        if df is not None and len(df) > 0:
            logger.info(f"✅ Existing dataset found with {len(df)} rows")
            logger.info("Dataset already available. Skipping download.")
            return
    except FileNotFoundError:
        logger.info("No existing dataset found. Proceeding with download...")
    
    # Kaggle dataset'lerini dene
    dataset_downloaded = False
    
    for dataset_key, dataset_info in GAMING_DATASETS.items():
        try:
            logger.info(f"\nTrying dataset: {dataset_info['name']}")
            logger.info(f"Description: {dataset_info['description']}")
            
            download_kaggle_dataset(
                dataset_info['name'],
                RAW_DATA_DIR
            )
            
            # İndirilen dataset'i kontrol et
            df = load_gaming_dataset(RAW_DATA_DIR)
            if df is not None and len(df) >= 10000:  # Minimum 10k rows requirement
                logger.info(f"✅ Dataset downloaded successfully!")
                logger.info(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                dataset_downloaded = True
                break
            else:
                logger.warning(f"Dataset too small ({len(df) if df is not None else 0} rows). Trying next...")
                
        except Exception as e:
            logger.warning(f"Could not download {dataset_key}: {e}")
            logger.info("Trying next dataset...")
            continue
    
    # Eğer hiçbir dataset indirilemediyse, sample dataset oluştur
    if not dataset_downloaded:
        logger.info("\n" + "=" * 60)
        logger.info("Could not download from Kaggle. Creating sample dataset...")
        logger.info("=" * 60)
        logger.info("\nTo use a real Kaggle dataset:")
        logger.info("1. Go to https://www.kaggle.com/datasets")
        logger.info("2. Search for 'gaming user behavior' or 'player segmentation'")
        logger.info("3. Download a dataset with:")
        logger.info("   - At least 10,000 rows")
        logger.info("   - At least 10 features")
        logger.info("   - Tabular format (CSV)")
        logger.info("4. Place the CSV file in data/raw/ directory")
        logger.info("\nFor now, creating a realistic sample dataset...")
        
        df = create_sample_gaming_dataset(n_samples=20000, save_path=RAW_DATA_DIR / "train.csv")
        logger.info(f"✅ Sample dataset created with {len(df)} rows and {len(df.columns)} features")
        logger.info("This dataset can be used for development and testing.")

if __name__ == "__main__":
    main()

