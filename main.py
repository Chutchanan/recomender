# main.py - Updated FastAPI application entry point
import asyncio
from contextlib import asynccontextmanager
from adapters.api import app
from adapters.database import init_db
from services.recommend_model import init_model_service
from config import settings

@asynccontextmanager
async def lifespan(app):
    """Application startup and shutdown"""
    # Startup
    print("🚀 Starting Restaurant Recommendation API v3.0...")
    
    try:
        # Initialize database
        print("📊 Initializing database connection...")
        await init_db()
        
        # Initialize model service
        print("🤖 Loading recommendation model...")
        await init_model_service(settings.MODEL_PATH)
        
        print("✅ All services initialized successfully!")
        print(f"🌐 API ready at http://{settings.API_HOST}:{settings.API_PORT}")
        print(f"📚 Documentation at http://{settings.API_HOST}:{settings.API_PORT}/docs")
        print("🔗 Main endpoint:")
        print(f"   - Predict: POST /predict/{{user_id}}")
        print("🔗 Other endpoints:")
        print(f"   - Health: GET /health")
        print(f"   - Model info: GET /model/info")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🔄 Shutting down services...")

# Set lifespan for the app
app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting server...")
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        workers=1,
        log_level="info"
    )