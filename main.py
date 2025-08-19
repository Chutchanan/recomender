# main.py - FastAPI application entry point
import asyncio
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from adapters.api import app
from adapters.database import init_db
from adapters.ml_model import init_model_service
from config import settings

@asynccontextmanager
async def lifespan(app):
    """Application startup and shutdown"""
    # Startup
    print("🚀 Starting Restaurant Recommendation API...")
    
    try:
        # Initialize database
        print("📊 Initializing database connection...")
        await init_db()
        
        # Initialize model service
        print("🤖 Loading ML model...")
        await init_model_service(settings.MODEL_PATH)
        
        print("✅ All services initialized successfully!")
        print(f"🌐 API ready at http://{settings.API_HOST}:{settings.API_PORT}")
        print(f"📚 Documentation at http://{settings.API_HOST}:{settings.API_PORT}/docs")
        
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
        reload=False,  # Set to True for development
        workers=1,     # Single worker since model is loaded in memory
        log_level="info"
    )