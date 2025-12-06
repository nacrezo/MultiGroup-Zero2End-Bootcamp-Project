"""
Configuration file for the Gaming User Segmentation project.
Contains paths, business rules, and model settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
# Gaming user segmentation için uygun dataset
# Örnek: "datasnaek/mobile-games-ab-testing" veya başka bir gaming dataset
DATASET_NAME = "gaming_user_segmentation"
TRAIN_FILE = RAW_DATA_DIR / "train.csv"
TEST_FILE = RAW_DATA_DIR / "test.csv"

# Target variable (user segmentation için cluster label)
TARGET_COLUMN = "user_segment"  # Bu clustering sonucu olacak

# Business rules - Gaming sektörü için
BUSINESS_RULES = {
    "segment_count": 4,  # Kullanıcı segment sayısı (K-Means için)
    "high_value_threshold": 100,  # Yüksek değerli kullanıcı eşiği (in-game purchase)
    "active_user_threshold": 7,  # Aktif kullanıcı için minimum gün sayısı
    "churn_definition": "User who has not played in the last 7 days",
    "retention_cost": 5,  # Kullanıcı tutma maliyeti (USD)
    "acquisition_cost": 10,  # Yeni kullanıcı kazanma maliyeti (USD)
    "min_silhouette_score": 0.3,  # Minimum silhouette score for good clustering
}

# Model configuration
MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "validation_size": 0.2,
    "cv_folds": 5,
    "scoring_metric": "silhouette_score",  # Clustering için silhouette score
    "n_clusters_range": (3, 8),  # Test edilecek cluster sayısı aralığı
}

# Feature engineering
FEATURE_CONFIG = {
    "categorical_encoding": "target",  # target, onehot, label
    "numerical_scaling": True,  # Clustering için scaling önemli
    "feature_selection": True,
    "top_n_features": 15,
    "remove_low_variance": True,
    "variance_threshold": 0.01,
}

# Clustering algorithms configuration
CLUSTERING_CONFIG = {
    "kmeans": {
        "n_clusters": 4,
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": MODEL_CONFIG["random_state"],
    },
    "dbscan": {
        "eps": 0.5,
        "min_samples": 5,
    },
    "hierarchical": {
        "n_clusters": 4,
        "linkage": "ward",
    },
}

# Model hyperparameters (if using supervised approach for validation)
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": MODEL_CONFIG["random_state"],
}

LIGHTGBM_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": MODEL_CONFIG["random_state"],
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

