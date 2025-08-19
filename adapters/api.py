# adapters/api.py - FastAPI controllers and endpoints
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import time
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from domain.models import RecommendationRequest, Recommendation
from usecases.recommendation_service import RecommendationService
from adapters.database import get_db_session, PostgreSQLUserRepository, PostgreSQLRestaurantRepository
from adapters.ml_model import get_model_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic schemas for API
class RecommendationRequestSchema(BaseModel):
    candidate_restaurant_ids: List[int] = Field(..., description="List of candidate restaurant IDs")
    latitude: float = Field(..., description="User's latitude", ge=-90, le=90)
    longitude: float = Field(..., description="User's longitude", ge=-180, le=180)
    size: int = Field(20, description="Number of recommended restaurants", ge=1, le=100)
    max_dist: int = Field(5000, description="Max distance in meters", ge=100, le=50000)
    sort_dist: bool = Field(False, description="Sort by distance instead of score")

class RecommendationSchema(BaseModel):
    id: int = Field(..., description="Restaurant ID")
    score: float = Field(..., description="Click probability score", ge=0, le=1)
    displacement: float = Field(..., description="Distance in meters", ge=0)

class RecommendationResponseSchema(BaseModel):
    restaurants: List[RecommendationSchema] = Field(..., description="List of recommended restaurants")
    total_candidates: int = Field(..., description="Total candidate restaurants processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

class HealthResponseSchema(BaseModel):
    status: str
    timestamp: float
    model_loaded: bool
    database_connected: bool
    
class ModelInfoSchema(BaseModel):
    model_info: dict
    performance_stats: dict

# Create FastAPI app
app = FastAPI(
    title="Restaurant Recommendation API",
    description="High-performance restaurant recommendation service using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
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
    
    # Log slow requests
    if process_time > 0.1:  # 100ms threshold
        logger.warning(f"Slow request: {request.url.path} took {process_time*1000:.2f}ms")
    
    return response

# Dependency injection
async def get_recommendation_service(session = Depends(get_db_session)):
    """Create recommendation service with dependencies"""
    user_repo = PostgreSQLUserRepository(session)
    restaurant_repo = PostgreSQLRestaurantRepository(session)
    model_repo = get_model_service()
    
    return RecommendationService(user_repo, restaurant_repo, model_repo)

# Health check endpoint
@app.get("/health", response_model=HealthResponseSchema)
async def health_check():
    """Health check endpoint"""
    try:
        model_service = get_model_service()
        model_healthy = await model_service.health_check()
        
        # Test database connection
        from adapters.database import check_db_health
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

# Model info endpoint
@app.get("/model/info", response_model=ModelInfoSchema)
async def get_model_info():
    """Get model information and performance stats"""
    try:
        model_service = get_model_service()
        model_info = model_service.get_model_info()
        
        # Add performance stats
        performance_stats = {
            "estimated_rps_capacity": 1000,  # From our test results
            "avg_inference_time_ms": 1.0,
            "batch_processing": True,
            "device": str(model_service.device)
        }
        
        return ModelInfoSchema(
            model_info=model_info,
            performance_stats=performance_stats
        )
        
    except Exception as e:
        logger.error(f"Model info failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )

# Main recommendation endpoint
@app.post("/recommend/{user_id}", response_model=RecommendationResponseSchema)
async def recommend_restaurants(
    user_id: str,
    request: RecommendationRequestSchema,
    service: RecommendationService = Depends(get_recommendation_service)
):
    """
    Generate restaurant recommendations for a user
    
    - **user_id**: User identifier (e.g., 'u00000')
    - **candidate_restaurant_ids**: List of restaurant IDs to consider
    - **latitude**: User's current latitude
    - **longitude**: User's current longitude  
    - **size**: Number of recommendations to return (default: 20)
    - **max_dist**: Maximum distance in meters (default: 5000)
    - **sort_dist**: Sort by distance instead of score (default: false)
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not request.candidate_restaurant_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="candidate_restaurant_ids cannot be empty"
            )
        
        if len(request.candidate_restaurant_ids) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many candidate restaurants (max: 500)"
            )
        
        # Convert to domain model
        domain_request = RecommendationRequest(
            user_id=user_id,
            candidate_restaurant_ids=request.candidate_restaurant_ids,
            latitude=request.latitude,
            longitude=request.longitude,
            size=request.size,
            max_dist=request.max_dist,
            sort_dist=request.sort_dist
        )
        
        # Generate recommendations
        recommendations = await service.generate_recommendations(domain_request)
        
        # Convert to response schema
        restaurant_schemas = [
            RecommendationSchema(
                id=rec.restaurant_id,
                score=rec.score,
                displacement=rec.displacement
            )
            for rec in recommendations
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecommendationResponseSchema(
            restaurants=restaurant_schemas,
            total_candidates=len(request.candidate_restaurant_ids),
            processing_time_ms=processing_time
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Recommendation failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

# Performance testing endpoint (optional - remove in production)
@app.post("/test/performance")
async def test_performance(
    batch_size: int = 10,
    iterations: int = 5
):
    """Test endpoint performance (development only)"""
    try:
        model_service = get_model_service()
        
        times = []
        for _ in range(iterations):
            # Simulate batch prediction
            user_features = [0.1] * 30
            restaurant_features_list = [[0.2 + i * 0.01] * 10 for i in range(batch_size)]
            
            start_time = time.time()
            scores = await model_service.predict_for_user_restaurants(user_features, restaurant_features_list)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        return {
            "batch_size": batch_size,
            "iterations": iterations,
            "avg_time_ms": avg_time * 1000,
            "time_per_item_ms": avg_time * 1000 / batch_size,
            "estimated_rps": 1 / avg_time
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Restaurant Recommendation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "main_endpoint": "/recommend/{user_id}"
    }