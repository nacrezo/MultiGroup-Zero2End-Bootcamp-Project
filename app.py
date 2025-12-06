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
    """User features for segmentation."""
    age: int = Field(..., ge=13, le=100, description="User age")
    gender: str = Field(..., description="User gender")
    country: str = Field(..., description="User country")
    device_type: str = Field(..., description="Device type")
    total_sessions: int = Field(..., ge=0, description="Total game sessions")
    total_playtime_hours: float = Field(..., ge=0, description="Total playtime in hours")
    total_spent_usd: float = Field(..., ge=0, description="Total spending in USD")
    login_frequency_per_week: float = Field(..., ge=0, description="Login frequency per week")
    max_level_reached: Optional[int] = Field(None, ge=0, description="Maximum level reached")
    levels_completed: Optional[int] = Field(None, ge=0, description="Levels completed")
    quests_completed: Optional[int] = Field(None, ge=0, description="Quests completed")
    achievements_unlocked: Optional[int] = Field(None, ge=0, description="Achievements unlocked")
    days_since_last_login: Optional[int] = Field(None, ge=0, description="Days since last login")
    days_since_registration: Optional[int] = Field(None, ge=0, description="Days since registration")
    friend_count: Optional[int] = Field(None, ge=0, description="Friend count")
    guild_member: Optional[int] = Field(None, ge=0, le=1, description="Guild membership")
    purchase_count: Optional[int] = Field(None, ge=0, description="Purchase count")
    premium_subscription: Optional[int] = Field(None, ge=0, le=1, description="Premium subscription")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Win rate")
    avg_score: Optional[float] = Field(None, ge=0, description="Average score")


class SegmentationResponse(BaseModel):
    """Response model for segmentation."""
    segment: int = Field(..., description="Predicted user segment (cluster ID)")
    segment_name: Optional[str] = Field(None, description="Human-readable segment name")
    confidence: Optional[float] = Field(None, description="Confidence score")


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
        segment = predict_user_segment(user_dict, pipeline)
        
        # Segment names (can be customized)
        segment_names = {
            0: "Casual Players",
            1: "Regular Players",
            2: "Engaged Players",
            3: "Whales (High Spenders)"
        }
        
        return SegmentationResponse(
            segment=segment,
            segment_name=segment_names.get(segment, f"Segment {segment}"),
            confidence=None  # Clustering doesn't provide confidence scores
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

