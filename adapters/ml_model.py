# adapters/ml_model.py - PyTorch model adapter for fast inference
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from domain.models import ModelRepository

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService(ModelRepository):
    """PyTorch model service for restaurant recommendations"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound model inference
        
    async def load_model(self):
        """Load the trained PyTorch model"""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from {self.model_path}")
            
            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, 
                self._load_model_sync
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Warm up with dummy prediction
            await self._warmup_model()
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        # Load the TorchScript model
        model = torch.jit.load(self.model_path, map_location=self.device)
        return model
    
    async def _warmup_model(self):
        """Warm up model with dummy data"""
        try:
            # Create dummy input (batch_size=1, features=40)
            dummy_input = torch.randn(1, 40, device=self.device)
            
            # Run inference in thread
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
    
    async def predict(self, user_features: List[float], restaurant_features: List[float]) -> float:
        """Single prediction for one user-restaurant pair"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Combine features (30 user + 10 restaurant = 40 features)
            combined_features = user_features + restaurant_features
            
            if len(combined_features) != 40:
                raise ValueError(f"Expected 40 features, got {len(combined_features)}")
            
            # Convert to tensor
            input_tensor = torch.tensor([combined_features], dtype=torch.float32, device=self.device)
            
            # Run prediction in thread to avoid blocking
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                input_tensor
            )
            
            # Apply sigmoid to get probability
            probability = torch.sigmoid(output).cpu().item()
            
            return probability
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def predict_batch(self, user_features_list: List[List[float]], 
                           restaurant_features_list: List[List[float]]) -> List[float]:
        """Batch prediction for multiple user-restaurant pairs"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if len(user_features_list) != len(restaurant_features_list):
            raise ValueError("User and restaurant feature lists must have same length")
        
        try:
            # Combine features for batch
            batch_features = []
            for user_feat, restaurant_feat in zip(user_features_list, restaurant_features_list):
                combined = user_feat + restaurant_feat
                if len(combined) != 40:
                    raise ValueError(f"Expected 40 features, got {len(combined)}")
                batch_features.append(combined)
            
            # Convert to tensor
            input_tensor = torch.tensor(batch_features, dtype=torch.float32, device=self.device)
            
            # Run batch prediction in thread
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                self.executor,
                self._predict_sync,
                input_tensor
            )
            
            # Apply sigmoid and convert to list
            probabilities = torch.sigmoid(output).cpu().numpy().tolist()
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    async def predict_for_user_restaurants(self, user_features: List[float], 
                                         restaurants_features: List[List[float]]) -> List[float]:
        """Optimized prediction for one user against multiple restaurants (implements ModelRepository interface)"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Prepare batch with same user features repeated
            user_features_list = [user_features] * len(restaurants_features)
            
            # Use batch prediction
            return await self.predict_batch(user_features_list, restaurants_features)
            
        except Exception as e:
            logger.error(f"User-restaurants prediction failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_loaded": self.model is not None,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "input_features": 40,
            "output": "click_probability"
        }
    
    async def health_check(self) -> bool:
        """Check if model is healthy"""
        try:
            if self.model is None:
                return False
            
            # Quick prediction test
            dummy_user = [0.0] * 30
            dummy_restaurant = [0.0] * 10
            await self.predict(dummy_user, dummy_restaurant)
            
            return True
            
        except Exception as e:
            logger.error(f"Model health check failed: {e}")
            return False

# Global model service instance
model_service: Optional[ModelService] = None

def get_model_service() -> ModelService:
    """Dependency injection for model service"""
    global model_service
    if model_service is None:
        raise RuntimeError("Model service not initialized")
    return model_service

async def init_model_service(model_path: str):
    """Initialize global model service"""
    global model_service
    model_service = ModelService(model_path)
    await model_service.load_model()
    return model_service