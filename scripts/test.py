# test_simple_load.py - Simple test to load data
import asyncio
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from adapters.database import Base, UserModel, RestaurantModel
from config import settings

DATA_DIR = Path("data")

async def simple_load_test():
    """Simple test to load a few users"""
    try:
        print("🔄 Simple load test...")
        
        # Create tables
        engine = create_async_engine(settings.DATABASE_URL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Test loading just 10 users
        user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
        print(f"📊 Found {len(user_df):,} users total")
        
        # Take only first 10 users for test
        test_users = user_df.head(10)
        
        SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession)
        async with SessionLocal() as session:
            # Clear existing data
            await session.execute(text("DELETE FROM users"))
            
            # Prepare test data
            users_data = []
            for _, row in test_users.iterrows():
                user_data = {
                    'user_id': row['user_id'],
                    'f00': float(row['f00']), 'f01': float(row['f01']), 'f02': float(row['f02']),
                    'f03': float(row['f03']), 'f04': float(row['f04']), 'f05': float(row['f05']),
                    'f06': float(row['f06']), 'f07': float(row['f07']), 'f08': float(row['f08']),
                    'f09': float(row['f09']), 'f10': float(row['f10']), 'f11': float(row['f11']),
                    'f12': float(row['f12']), 'f13': float(row['f13']), 'f14': float(row['f14']),
                    'f15': float(row['f15']), 'f16': float(row['f16']), 'f17': float(row['f17']),
                    'f18': float(row['f18']), 'f19': float(row['f19']), 'f20': float(row['f20']),
                    'f21': float(row['f21']), 'f22': float(row['f22']), 'f23': float(row['f23']),
                    'f24': float(row['f24']), 'f25': float(row['f25']), 'f26': float(row['f26']),
                    'f27': float(row['f27']), 'f28': float(row['f28']), 'f29': float(row['f29'])
                }
                users_data.append(user_data)
            
            # Insert test data
            await session.execute(UserModel.__table__.insert(), users_data)
            await session.commit()
            
            # Verify
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            count = result.scalar()
            
            print(f"✅ Successfully loaded {count} test users")
        
        await engine.dispose()
        print("🎉 Simple test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(simple_load_test())