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
from sklearn.impute import SimpleImputer
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
        self.imputer = SimpleImputer(strategy='median')
        self.label_encoders = {}
        self.variance_selector = None
        self.model = None
        self.feature_names = None
        self.n_clusters = BUSINESS_RULES['segment_count']
        self.is_fitted = False
        
    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Preprocess the data for clustering."""
        df = df.copy()
        
        # Drop non-feature columns
        cols_to_drop = ['PlayerID', 'EngagementLevel'] 
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        
        # Categorical columns to encode
        cat_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
        
        # One-Hot Encoding
        for col in cat_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False, dtype=int)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[col], inplace=True)
                
                
                # Note: Alignment happens after all encoding is done
        
        # Align with training features if fitted
        if self.is_fitted:
             # Align with training features: add missing with 0
             for feat in self.feature_names:
                 if feat not in df.columns:
                     df[feat] = 0
                     
        # No specific derived features for now, basic clustering on raw+encoded data
        
        # Check for duplicates (Verification)
        if df.columns.duplicated().any():
            dups = df.columns[df.columns.duplicated()].tolist()
            # If duplicates exist, we should probably resolve them (e.g., keep first)
            # But with correct logic, they shouldn't exist.
            # If they do, it's a bug. But let's be robust.
            df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    
    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the clustering pipeline."""
        logger.info("Starting pipeline training...")
        
        # Preprocess
        df_processed = self.preprocess(df, is_training=True)
        
        # Check for duplicates
        if df_processed.columns.duplicated().any():
            dups = df_processed.columns[df_processed.columns.duplicated()].tolist()
            logger.error(f"Duplicate columns found: {dups}")
            raise ValueError(f"Duplicate columns: {dups}")
        
        # Select numerical features
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        X = df_processed[numerical_cols].copy()
        self.feature_names = X.columns.tolist()
        
        # Remove low variance features
        if FEATURE_CONFIG['remove_low_variance']:
            self.variance_selector = VarianceThreshold(threshold=FEATURE_CONFIG['variance_threshold'])
            # Impute before variance threshold if needed, but usually VarianceThreshold handles NaNs? No it doesn't.
            # So impute first.
            X_imputed = self.imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            # Now variance threshold
            X = pd.DataFrame(
                self.variance_selector.fit_transform(X),
                columns=[self.feature_names[i] for i in range(len(self.feature_names)) 
                        if self.variance_selector.variances_[i] >= FEATURE_CONFIG['variance_threshold']],
                index=X.index
            )
            self.feature_names = X.columns.tolist()
        else:
            # Impute even if no variance selection
            X_imputed = self.imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
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
        
        self.is_fitted = True
        
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
        # Ensure all columns exist (preprocess should have handled dtypes, but we double check or reindex)
        # Reindex is safer to ensure order and missing/extra columns
        X = df_processed.reindex(columns=self.feature_names, fill_value=0)
        
        # Remove low variance (if selector was used)
        if self.variance_selector is not None:
             # Impute first
            X_imputed = self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            X = pd.DataFrame(
                self.variance_selector.transform(X),
                columns=self.feature_names,
                index=X.index
            )
        else:
             # Impute
            X_imputed = self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
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
            'imputer': self.imputer,
            'label_encoders': self.label_encoders,
            'variance_selector': self.variance_selector,
            'model': self.model,
            'feature_names': self.feature_names,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted
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
        self.imputer = pipeline_dict.get('imputer', SimpleImputer(strategy='median')) # Backwards compat
        self.label_encoders = pipeline_dict['label_encoders']
        self.variance_selector = pipeline_dict['variance_selector']
        self.model = pipeline_dict['model']
        self.feature_names = pipeline_dict['feature_names']
        self.n_clusters = pipeline_dict['n_clusters']
        self.is_fitted = pipeline_dict.get('is_fitted', True) # Default to True for backward compat if needed
        
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
        raise FileNotFoundError(
            "Dataset not found. Please download the dataset manually from Kaggle:\n"
            "https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset\n"
            "Place the CSV file in data/raw/ directory."
        )
    
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

