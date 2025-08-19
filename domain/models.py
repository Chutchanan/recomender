# domain/models.py - Core business entities
from dataclasses import dataclass
from typing import List, Optional
from abc import ABC, abstractmethod

@dataclass
class User:
    id: str
    features: List[float]  # f00-f29 (30 features)
    
    def __post_init__(self):
        if len(self.features) != 30:
            raise ValueError(f"User features must have 30 elements, got {len(self.features)}")

@dataclass 
class Restaurant:
    id: int
    features: List[float]  # f00-f09 (10 features)
    latitude: float
    longitude: float
    
    def __post_init__(self):
        if len(self.features) != 10:
            raise ValueError(f"Restaurant features must have 10 elements, got {len(self.features)}")

@dataclass
class Recommendation:
    restaurant_id: int
    score: float  # Click probability from model
    displacement: float  # Distance in meters
    
@dataclass
class RecommendationRequest:
    user_id: str
    candidate_restaurant_ids: List[int]
    latitude: float
    longitude: float
    size: int = 20
    max_dist: int = 5000
    sort_dist: bool = False

# Repository interfaces (Uncle Bob's dependency inversion)
class UserRepository(ABC):
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID with features"""
        pass

class RestaurantRepository(ABC):
    @abstractmethod 
    async def get_restaurants(self, restaurant_ids: List[int]) -> List[Restaurant]:
        """Get restaurants by IDs with features and locations"""
        pass
    
    @abstractmethod
    async def get_restaurants_in_radius(self, lat: float, lon: float, radius_m: int) -> List[Restaurant]:
        """Get restaurants within radius using spatial query"""
        pass

class ModelRepository(ABC):
    @abstractmethod
    async def predict(self, user_features: List[float], restaurant_features: List[float]) -> float:
        """Get click probability prediction"""
        pass
    
    @abstractmethod
    async def predict_for_user_restaurants(self, user_features: List[float], 
                                         restaurants_features: List[List[float]]) -> List[float]:
        """Optimized prediction for one user against multiple restaurants"""
        pass