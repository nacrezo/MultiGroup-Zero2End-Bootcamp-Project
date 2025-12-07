"""
FastAPI application for Gaming User Segmentation.
Provides REST API endpoints for user segment prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import pandas as pd

from src.inference import predict_user_segment, predict_batch, get_segment_profile, UserSegmentationPipeline
from src.config import API_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gaming User Segmentation API",
    description="API for segmenting gaming users based on behavior and engagement",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
try:
    pipeline = UserSegmentationPipeline()
    pipeline.load()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    pipeline = None


# Request/Response models
class UserFeatures(BaseModel):
    """User features for segmentation (Gaming Behavior Dataset)."""
    PlayerID: Optional[int] = Field(None, description="Player ID")
    Age: int = Field(..., ge=13, le=99, description="Player Age")
    Gender: str = Field(..., description="Gender")
    Location: str = Field(..., description="Location")
    GameGenre: str = Field(..., description="Game Genre")
    PlayTimeHours: float = Field(..., ge=0, description="Play Time Hours")
    InGamePurchases: int = Field(..., ge=0, le=1, description="In Game Purchases (0 or 1)")
    GameDifficulty: str = Field(..., description="Game Difficulty")
    SessionsPerWeek: int = Field(..., ge=0, description="Sessions Per Week")
    AvgSessionDurationMinutes: float = Field(..., ge=0, description="Avg Session Duration")
    PlayerLevel: int = Field(..., ge=1, description="Player Level")
    AchievementUnlocked: int = Field(..., ge=0, description="Achievements Unlocked")


class SegmentationResponse(BaseModel):
    cluster: int = Field(..., description="Predicted user segment (cluster ID)")
    segment_name: str = Field(..., description="Human-readable segment name")
    message: str = Field(..., description="A descriptive message about the segmentation result.")


class BatchSegmentationRequest(BaseModel):
    """Request model for batch segmentation."""
    users: List[Dict[str, Any]] = Field(..., description="List of user feature dictionaries")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Gaming User Segmentation API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict segment for a single user",
            "/predict/batch": "POST - Predict segments for multiple users",
            "/segment/{segment_id}": "GET - Get segment profile",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if pipeline is not None else "unhealthy",
        "model_loaded": pipeline is not None
    }


@app.post("/predict", response_model=SegmentationResponse)
async def predict_segment(user: UserFeatures):
    """Predict user segment."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dictionary
        user_dict = user.dict(exclude_none=True)
        
        # Predict
        cluster = predict_user_segment(user_dict, pipeline)
        
        segment_names = {
            0: "Action & Sports Fans",
            1: "Simulation Enthusiasts",
            2: "Strategy Masterminds",
            3: "RPG Adventurers"
        }
        
        return SegmentationResponse(
            cluster=cluster,
            segment_name=segment_names.get(cluster, "Unknown"),
            message=f"User belongs to segment {cluster}: {segment_names.get(cluster, 'Unknown')}"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch_segments(request: BatchSegmentationRequest):
    """Predict segments for multiple users."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.users)
        
        # Predict
        segments = predict_batch(df, pipeline)
        
        return {
            "segments": segments.tolist(),
            "count": len(segments)
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segment/{segment_id}")
async def get_segment(segment_id: int):
    """Get segment profile and statistics."""
    try:
        profile = get_segment_profile(segment_id)
        return profile
    except Exception as e:
        logger.error(f"Error getting segment profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import pandas as pd
    uvicorn.run(
        "app:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )

