"""
Final ML Pipeline for Gaming User Segmentation.
This script contains the complete preprocessing and training pipeline for clustering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import logging
from typing import Tuple, Dict, Any, Optional

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_selection import VarianceThreshold

import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import *
from src.data_loader import load_gaming_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserSegmentationPipeline:
    """Complete pipeline for user segmentation."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.variance_selector = None
        self.model = None
        self.feature_names = None
        self.n_clusters = BUSINESS_RULES['segment_count']
        
    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess the data for clustering."""
        df = df.copy()
        
        # Remove non-feature columns
        cols_to_remove = ['user_id']
        if 'registration_date' in df.columns:
            cols_to_remove.append('registration_date')
        df = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str).fillna('Unknown'))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    df[col] = df[col].astype(str).fillna('Unknown')
                    unseen_mask = ~df[col].isin(self.label_encoders[col].classes_)
                    if unseen_mask.any():
                        df.loc[unseen_mask, col] = self.label_encoders[col].classes_[0]
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                else:
                    df[f'{col}_encoded'] = 0
        
        # Feature engineering
        if 'total_spent_usd' in df.columns and 'total_playtime_hours' in df.columns:
            df['spending_per_hour'] = df['total_spent_usd'] / (df['total_playtime_hours'] + 1)
        
        if 'total_sessions' in df.columns and 'days_since_registration' in df.columns:
            df['sessions_per_week'] = df['total_sessions'] / ((df['days_since_registration'] / 7) + 1)
        
        if 'levels_completed' in df.columns and 'max_level_reached' in df.columns:
            df['level_progress_rate'] = df['levels_completed'] / (df['max_level_reached'] + 1)
        
        # Aggregate features
        if 'total_sessions' in df.columns and 'total_playtime_hours' in df.columns:
            df['total_gameplay_score'] = (df['total_sessions'] * 0.3 + 
                                         df['total_playtime_hours'] * 0.3 + 
                                         df.get('max_level_reached', 0) * 0.2 + 
                                         df.get('quests_completed', 0) * 0.2)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the clustering pipeline."""
        logger.info("Starting pipeline training...")
        
        # Preprocess
        df_processed = self.preprocess(df, is_training=True)
        
        # Select numerical features
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[numerical_cols].copy()
        self.feature_names = X.columns.tolist()
        
        # Remove low variance features
        if FEATURE_CONFIG['remove_low_variance']:
            self.variance_selector = VarianceThreshold(threshold=FEATURE_CONFIG['variance_threshold'])
            X = pd.DataFrame(
                self.variance_selector.fit_transform(X),
                columns=[self.feature_names[i] for i in range(len(self.feature_names)) 
                        if self.variance_selector.variances_[i] >= FEATURE_CONFIG['variance_threshold']],
                index=X.index
            )
            self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train clustering model
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=MODEL_CONFIG['random_state']
        )
        self.model.fit(X_scaled)
        
        # Calculate metrics
        labels = self.model.labels_
        metrics = {
            'silhouette_score': silhouette_score(X_scaled, labels),
            'davies_bouldin_index': davies_bouldin_score(X_scaled, labels),
            'calinski_harabasz_index': calinski_harabasz_score(X_scaled, labels),
            'inertia': self.model.inertia_,
            'n_clusters': self.n_clusters,
            'n_features': len(self.feature_names)
        }
        
        logger.info(f"Pipeline trained successfully!")
        logger.info(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        logger.info(f"Features used: {len(self.feature_names)}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict clusters for new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Preprocess
        df_processed = self.preprocess(df, is_training=False)
        
        # Select features
        X = df_processed[self.feature_names].copy()
        
        # Remove low variance (if selector was used)
        if self.variance_selector is not None:
            X = pd.DataFrame(
                self.variance_selector.transform(X),
                columns=self.feature_names,
                index=X.index
            )
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        labels = self.model.predict(X_scaled)
        
        return labels
    
    def save(self, model_path: Path = None) -> None:
        """Save the pipeline."""
        if model_path is None:
            model_path = MODELS_DIR / "user_segmentation_pipeline.pkl"
        
        pipeline_dict = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'variance_selector': self.variance_selector,
            'model': self.model,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters
        }
        
        joblib.dump(pipeline_dict, model_path)
        logger.info(f"Pipeline saved to {model_path}")
    
    def load(self, model_path: Path = None) -> None:
        """Load the pipeline."""
        if model_path is None:
            model_path = MODELS_DIR / "user_segmentation_pipeline.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        pipeline_dict = joblib.load(model_path)
        self.scaler = pipeline_dict['scaler']
        self.label_encoders = pipeline_dict['label_encoders']
        self.variance_selector = pipeline_dict['variance_selector']
        self.model = pipeline_dict['model']
        self.feature_names = pipeline_dict['feature_names']
        self.n_clusters = pipeline_dict['n_clusters']
        
        logger.info(f"Pipeline loaded from {model_path}")


def train_pipeline() -> Dict[str, Any]:
    """Train the complete pipeline."""
    logger.info("=" * 60)
    logger.info("GAMING USER SEGMENTATION PIPELINE")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading dataset...")
    df = load_gaming_dataset(RAW_DATA_DIR)
    if df is None or len(df) == 0:
        raise FileNotFoundError("Dataset not found. Please run download_dataset.py first.")
    
    logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create and train pipeline
    pipeline = UserSegmentationPipeline()
    metrics = pipeline.fit(df)
    
    # Save pipeline
    pipeline.save()
    
    # Save processed data
    df_processed = pipeline.preprocess(df, is_training=True)
    df_processed['cluster'] = pipeline.predict(df)
    processed_path = PROCESSED_DATA_DIR / "segmented_users.csv"
    df_processed.to_csv(processed_path, index=False)
    logger.info(f"Processed data saved to {processed_path}")
    
    return metrics


if __name__ == "__main__":
    metrics = train_pipeline()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    for key, value in metrics.items():
        print(f"{key}: {value}")

