# services/recommend_model.py - Model Service for PostgreSQL predictions
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.ext.asyncio import AsyncSession
from adapters.database import get_user, get_restaurants

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendModelService:
    """Model service that uses PostgreSQL data + model.pt for predictions"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_model(self):
        """Load the trained PyTorch model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, 
                self._load_model_sync
            )
            
            self.model.eval()
            await self._warmup_model()
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        model = torch.jit.load(self.model_path, map_location=self.device)
        return model
    
    async def _warmup_model(self):
        """Warm up model with dummy data"""
        try:
            dummy_input = torch.randn(1, 40, device=self.device)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                dummy_input
            )
            logger.info("🔥 Model warmed up successfully")
        except Exception as e:
            logger.warning(f"⚠️  Model warmup failed: {e}")
    
    def _predict_sync(self, input_tensor):
        """Synchronous prediction"""
        with torch.no_grad():
            return self.model(input_tensor)
    
    async def predict_restaurants_for_user(self, session: AsyncSession, user_id: str, 
                                         candidate_restaurant_ids: List[int]) -> List[dict]:
        """
        Main prediction method: Get user and restaurants from PostgreSQL, 
        then predict scores using model.pt
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Get user from PostgreSQL
            user = await get_user(session, user_id)
            if not user:
                raise ValueError(f"User {user_id} not found in database")
            
            # Get restaurants from PostgreSQL
            restaurants = await get_restaurants(session, candidate_restaurant_ids)
            if not restaurants:
                raise ValueError("No restaurants found in database")
            
            # Prepare batch features (like standalone script)
            batch_features = []
            valid_restaurants = []
            
            for restaurant in restaurants:
                # Combine user features (30) + restaurant features (10) = 40 total
                combined_features = user.features + restaurant.features
                if len(combined_features) != 40:
                    logger.warning(f"Skipping restaurant {restaurant.id}: invalid feature count")
                    continue
                    
                batch_features.append(combined_features)
                valid_restaurants.append(restaurant)
            
            if not batch_features:
                raise ValueError("No valid restaurant features found")
            
            # Convert to tensor
            input_tensor = torch.tensor(batch_features, dtype=torch.float32, device=self.device)
            
            # Run batch prediction
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                input_tensor
            )
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(output).cpu().numpy()
            
            # Format results
            results = []
            for restaurant, prob in zip(valid_restaurants, probabilities):
                results.append({
                    "restaurant_id": restaurant.id,
                    "score": float(prob),
                    "latitude": restaurant.latitude,
                    "longitude": restaurant.longitude
                })
            
            # Sort by score (highest first)
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction failed for user {user_id}: {e}")
            raise
    
    async def predict_restaurants_for_user_with_location(self, session: AsyncSession, user_id: str, 
                                                       candidate_restaurant_ids: List[int],
                                                       user_lat: float, user_lon: float,
                                                       max_dist: int, size: int, sort_dist: bool) -> List[dict]:
        """
        Prediction with location filtering and sorting
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Get user from PostgreSQL
            user = await get_user(session, user_id)
            if not user:
                raise ValueError(f"User {user_id} not found in database")
            
            # Get restaurants from PostgreSQL
            restaurants = await get_restaurants(session, candidate_restaurant_ids)
            if not restaurants:
                raise ValueError("No restaurants found in database")
            
            # Filter by distance and prepare for ML prediction
            valid_restaurants = []
            valid_features = []
            
            for restaurant in restaurants:
                # Calculate distance using Haversine formula
                distance = self._calculate_distance(user_lat, user_lon, restaurant.latitude, restaurant.longitude)
                
                if distance <= max_dist:
                    valid_restaurants.append((restaurant, distance))
                    valid_features.append(user.features + restaurant.features)
            
            if not valid_restaurants:
                return []  # No restaurants within distance
            
            # Batch ML prediction
            input_tensor = torch.tensor(valid_features, dtype=torch.float32, device=self.device)
            
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(self.executor, self._predict_sync, input_tensor)
            
            # Apply sigmoid to get probabilities
            probabilities = torch.sigmoid(output).cpu().numpy()
            
            # Format results
            results = []
            for (restaurant, distance), prob in zip(valid_restaurants, probabilities):
                results.append({
                    "restaurant_id": restaurant.id,
                    "score": float(prob),
                    "latitude": restaurant.latitude,
                    "longitude": restaurant.longitude,
                    "displacement": float(distance)
                })
            
            # Sort by distance or score
            if sort_dist:
                results.sort(key=lambda x: x["displacement"])
            else:
                results.sort(key=lambda x: x["score"], reverse=True)
            
            # Limit to requested size
            return results[:size]
            
        except Exception as e:
            logger.error(f"Location-based prediction failed for user {user_id}: {e}")
            raise
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in meters using Haversine formula"""
        import math
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        r = 6371000
        return c * r
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_loaded": self.model is not None,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "input_features": 40,
            "output": "click_probability",
            "data_source": "postgresql"
        }
    
    async def health_check(self) -> bool:
        """Check if model is healthy"""
        try:
            if self.model is None:
                return False
            
            # Quick prediction test with dummy data
            dummy_input = torch.randn(1, 40, device=self.device)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._predict_sync, dummy_input)
            
            return True
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False

# Global model service instance
model_service: Optional[RecommendModelService] = None

def get_model_service() -> RecommendModelService:
    """Dependency injection for model service"""
    global model_service
    if model_service is None:
        raise RuntimeError("Model service not initialized")
    return model_service

async def init_model_service(model_path: str):
    """Initialize global model service"""
    global model_service
    model_service = RecommendModelService(model_path)
    await model_service.load_model()
    return model_service