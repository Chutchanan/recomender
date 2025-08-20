# adapters/api.py - Simplified FastAPI with single prediction endpoint
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import time
import logging

from services.recommend_model import get_model_service
from adapters.database import get_db_session, check_db_health

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic schemas
class RecommendationRequest(BaseModel):
    candidate_restaurant_ids: List[int] = Field(..., description="List of candidate restaurant IDs", min_items=1, max_items=1500)
    latitude: float = Field(..., description="User's latitude", ge=-90, le=90)
    longitude: float = Field(..., description="User's longitude", ge=-180, le=180)
    size: int = Field(..., description="Number of recommended restaurants", ge=1, le=500)
    max_dist: int = Field(..., description="Max distance in meters", ge=100, le=100000)
    sort_dist: bool = Field(False, description="Sort by distance instead of score")

class RecommendationResponse(BaseModel):
    restaurant_id: int = Field(..., description="Restaurant ID")
    score: float = Field(..., description="Click probability score", ge=0, le=1)
    latitude: float = Field(..., description="Restaurant latitude")
    longitude: float = Field(..., description="Restaurant longitude")
    displacement: float = Field(..., description="Distance in meters", ge=0)

class PredictionResult(BaseModel):
    user_id: str = Field(..., description="User ID")
    recommendations: List[RecommendationResponse] = Field(..., description="List of recommendations")
    total_candidates: int = Field(..., description="Total candidate restaurants")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class HealthResponseSchema(BaseModel):
    status: str
    timestamp: float
    model_loaded: bool
    database_connected: bool

# Create FastAPI app
app = FastAPI(
    title="Restaurant Recommendation API",
    description="Simple restaurant recommendation service using PostgreSQL + model.pt",
    version="3.0.0"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Performance monitoring middleware
@app.middleware("http")
async def performance_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    if process_time > 0.1:
        logger.warning(f"Slow request: {request.url.path} took {process_time*1000:.2f}ms")
    
    return response

# Health check endpoint
@app.get("/health", response_model=HealthResponseSchema)
async def health_check():
    """Health check endpoint"""
    try:
        model_service = get_model_service()
        model_healthy = await model_service.health_check()
        db_healthy = await check_db_health()
        
        return HealthResponseSchema(
            status="healthy" if (model_healthy and db_healthy) else "unhealthy",
            timestamp=time.time(),
            model_loaded=model_healthy,
            database_connected=db_healthy
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unhealthy"
        )

# Main recommendation endpoint
@app.post("/recommend/{user_id}", response_model=PredictionResult)
async def recommend_restaurants(
    user_id: str,
    request: RecommendationRequest
):
    """
    Get restaurant recommendations using PostgreSQL data + model.pt
    
    - **user_id**: User identifier (e.g., '0', '1', 'u00000')
    - **candidate_restaurant_ids**: List of restaurant IDs to score (1-1500 items)
    - **latitude**: User's current latitude
    - **longitude**: User's current longitude  
    - **size**: Number of recommendations to return (required)
    - **max_dist**: Maximum distance in meters (required)
    - **sort_dist**: Sort by distance instead of score (default: false)
    
    Returns recommendations sorted by predicted click probability (highest first) or distance
    """
    start_time = time.time()
    
    try:
        # Get database session using Depends injection
        session = get_db_session()
        async with session.__anext__() as db_session:
            # Get model service
            model_service = get_model_service()
            
            # Get predictions with location filtering
            results = await model_service.predict_restaurants_for_user_with_location(
                db_session, user_id, request.candidate_restaurant_ids,
                request.latitude, request.longitude, 
                request.max_dist, request.size, request.sort_dist
            )
            
            # Format response
            recommendations = [
                RecommendationResponse(
                    restaurant_id=result["restaurant_id"],
                    score=result["score"],
                    latitude=result["latitude"],
                    longitude=result["longitude"],
                    displacement=result["displacement"]
                )
                for result in results
            ]
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResult(
                user_id=user_id,
                recommendations=recommendations,
                total_candidates=len(request.candidate_restaurant_ids),
                processing_time_ms=processing_time
            )
            
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    try:
        model_service = get_model_service()
        return model_service.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Restaurant Recommendation API",
        "version": "3.0.0",
        "description": "PostgreSQL + model.pt predictions",
        "main_endpoint": "POST /recommend/{user_id}",
        "health_check": "GET /health",
        "documentation": "GET /docs",
        "example_request": {
            "method": "POST",
            "url": "/recommend/0",
            "body": {
                "candidate_restaurant_ids": [1, 2, 3, 4, 5]
            }
        }
    }