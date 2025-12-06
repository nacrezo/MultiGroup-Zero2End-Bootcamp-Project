
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.append(str(Path.cwd()))

from src.pipeline import UserSegmentationPipeline

def analyze_clusters():
    # Load pipeline
    pipeline = UserSegmentationPipeline()
    pipeline.load()
    
    print("Pipeline loaded.")
    print(f"Features: {pipeline.feature_names}")
    
    # Get cluster centers (in scaled space)
    centers_scaled = pipeline.model.cluster_centers_
    scaler = pipeline.scaler
    centers_real = scaler.inverse_transform(centers_scaled)
    
    print("\nCluster Profiles (Key Metrics):")
    key_features = ['PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 'PlayerLevel', 'AvgSessionDurationMinutes']
    
    # Create indices for key features
    indices = [pipeline.feature_names.index(f) for f in key_features if f in pipeline.feature_names]
    
    centers_key = centers_real[:, indices]
    
    print("\nCluster Profiles (All Features):")
    
    # Print all feature centers to find any difference
    for i in range(len(centers_real)):
        print(f"\nCluster {i}:")
        for j, feature in enumerate(pipeline.feature_names):
            # Only print if value is significant (> 0.1) to reduce noise
            if abs(centers_real[i][j]) > 0.01:
                print(f"  {feature}: {centers_real[i][j]:.2f}")

if __name__ == "__main__":
    analyze_clusters()
