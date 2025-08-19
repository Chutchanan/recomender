# usecases/recommendation_service.py - Business logic (Uncle Bob's use cases layer)
import math
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from domain.models import (
    User, Restaurant, Recommendation, RecommendationRequest,
    UserRepository, RestaurantRepository, ModelRepository
)

class RecommendationService:
    def __init__(
        self,
        user_repo: UserRepository,
        restaurant_repo: RestaurantRepository, 
        model_repo: ModelRepository
    ):
        self.user_repo = user_repo
        self.restaurant_repo = restaurant_repo
        self.model_repo = model_repo
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in meters using Haversine formula"""
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
    
    async def generate_recommendations(self, request: RecommendationRequest) -> List[Recommendation]:
        """Core business logic for generating recommendations"""
        
        # 1. Get user data
        user = await self.user_repo.get_user(request.user_id)
        if not user:
            raise ValueError(f"User {request.user_id} not found")
        
        # 2. Get candidate restaurants
        restaurants = await self.restaurant_repo.get_restaurants(request.candidate_restaurant_ids)
        if not restaurants:
            raise ValueError("No restaurants found for given IDs")
        
        # 3. Filter by distance and collect valid restaurants
        valid_restaurants = []
        valid_restaurant_features = []
        
        for restaurant in restaurants:
            distance = self.calculate_distance(
                request.latitude, request.longitude,
                restaurant.latitude, restaurant.longitude
            )
            if distance <= request.max_dist:
                valid_restaurants.append((restaurant, distance))
                valid_restaurant_features.append(restaurant.features)
        
        if not valid_restaurants:
            return []  # No restaurants within distance
        
        # 4. Get batch predictions for all valid restaurants (PERFORMANCE OPTIMIZATION)
        scores = await self.model_repo.predict_for_user_restaurants(
            user.features, 
            valid_restaurant_features
        )
        
        # 5. Create recommendations
        recommendations = []
        for (restaurant, distance), score in zip(valid_restaurants, scores):
            recommendations.append(Recommendation(
                restaurant_id=restaurant.id,
                score=float(score),
                displacement=float(distance)
            ))
        
        # 6. Sort by score or distance
        if request.sort_dist:
            recommendations.sort(key=lambda x: x.displacement)
        else:
            recommendations.sort(key=lambda x: x.score, reverse=True)
        
        # 7. Limit to requested size
        return recommendations[:request.size]