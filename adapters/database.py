# adapters/database.py - PostgreSQL implementation with Domain Models
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, Float, text, select
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from config import settings

# Domain Models (moved from ml_model.py)
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

# SQLAlchemy Models
class Base(DeclarativeBase):
    pass

class UserModel(Base):
    __tablename__ = "users"
    
    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Individual feature columns (f00-f29)
    f00: Mapped[float] = mapped_column(Float, nullable=False)
    f01: Mapped[float] = mapped_column(Float, nullable=False)
    f02: Mapped[float] = mapped_column(Float, nullable=False)
    f03: Mapped[float] = mapped_column(Float, nullable=False)
    f04: Mapped[float] = mapped_column(Float, nullable=False)
    f05: Mapped[float] = mapped_column(Float, nullable=False)
    f06: Mapped[float] = mapped_column(Float, nullable=False)
    f07: Mapped[float] = mapped_column(Float, nullable=False)
    f08: Mapped[float] = mapped_column(Float, nullable=False)
    f09: Mapped[float] = mapped_column(Float, nullable=False)
    f10: Mapped[float] = mapped_column(Float, nullable=False)
    f11: Mapped[float] = mapped_column(Float, nullable=False)
    f12: Mapped[float] = mapped_column(Float, nullable=False)
    f13: Mapped[float] = mapped_column(Float, nullable=False)
    f14: Mapped[float] = mapped_column(Float, nullable=False)
    f15: Mapped[float] = mapped_column(Float, nullable=False)
    f16: Mapped[float] = mapped_column(Float, nullable=False)
    f17: Mapped[float] = mapped_column(Float, nullable=False)
    f18: Mapped[float] = mapped_column(Float, nullable=False)
    f19: Mapped[float] = mapped_column(Float, nullable=False)
    f20: Mapped[float] = mapped_column(Float, nullable=False)
    f21: Mapped[float] = mapped_column(Float, nullable=False)
    f22: Mapped[float] = mapped_column(Float, nullable=False)
    f23: Mapped[float] = mapped_column(Float, nullable=False)
    f24: Mapped[float] = mapped_column(Float, nullable=False)
    f25: Mapped[float] = mapped_column(Float, nullable=False)
    f26: Mapped[float] = mapped_column(Float, nullable=False)
    f27: Mapped[float] = mapped_column(Float, nullable=False)
    f28: Mapped[float] = mapped_column(Float, nullable=False)
    f29: Mapped[float] = mapped_column(Float, nullable=False)
    
    def to_domain(self) -> User:
        features = [
            self.f00, self.f01, self.f02, self.f03, self.f04, self.f05, self.f06, self.f07, self.f08, self.f09,
            self.f10, self.f11, self.f12, self.f13, self.f14, self.f15, self.f16, self.f17, self.f18, self.f19,
            self.f20, self.f21, self.f22, self.f23, self.f24, self.f25, self.f26, self.f27, self.f28, self.f29
        ]
        return User(id=self.user_id, features=features)

class RestaurantModel(Base):
    __tablename__ = "restaurants"
    
    restaurant_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    # Individual feature columns (f00-f09)
    f00: Mapped[float] = mapped_column(Float, nullable=False)
    f01: Mapped[float] = mapped_column(Float, nullable=False)
    f02: Mapped[float] = mapped_column(Float, nullable=False)
    f03: Mapped[float] = mapped_column(Float, nullable=False)
    f04: Mapped[float] = mapped_column(Float, nullable=False)
    f05: Mapped[float] = mapped_column(Float, nullable=False)
    f06: Mapped[float] = mapped_column(Float, nullable=False)
    f07: Mapped[float] = mapped_column(Float, nullable=False)
    f08: Mapped[float] = mapped_column(Float, nullable=False)
    f09: Mapped[float] = mapped_column(Float, nullable=False)
    
    def to_domain(self) -> Restaurant:
        features = [self.f00, self.f01, self.f02, self.f03, self.f04, self.f05, self.f06, self.f07, self.f08, self.f09]
        return Restaurant(
            id=self.restaurant_id,
            features=features,
            latitude=self.latitude,
            longitude=self.longitude
        )

# Database Engine and Session
engine = None
SessionLocal = None

async def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal
    
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
        pool_pre_ping=True
    )
    
    SessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("✅ Database initialized successfully")

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection for database sessions"""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Repository Functions
async def get_user(session: AsyncSession, user_id: str) -> Optional[User]:
    """Get user by ID"""
    stmt = select(UserModel).where(UserModel.user_id == user_id)
    result = await session.execute(stmt)
    user_model = result.scalar_one_or_none()
    return user_model.to_domain() if user_model else None

async def get_restaurants(session: AsyncSession, restaurant_ids: List[int]) -> List[Restaurant]:
    """Get restaurants by IDs"""
    stmt = select(RestaurantModel).where(RestaurantModel.restaurant_id.in_(restaurant_ids))
    result = await session.execute(stmt)
    restaurant_models = result.scalars().all()
    return [model.to_domain() for model in restaurant_models]

# Health check function
async def check_db_health() -> bool:
    """Check if database is healthy"""
    try:
        async with SessionLocal() as session:
            await session.execute(text("SELECT 1"))
            return True
    except Exception as e:
        print(f"Database health check failed: {e}")
        return False