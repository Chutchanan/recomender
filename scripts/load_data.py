# scripts/load_data.py - Fixed PostgreSQL data loader
import pandas as pd
import psycopg2
from pathlib import Path
import time
import numpy as np

DATA_DIR = Path("data")

def connect_to_database():
    """Connect to PostgreSQL database (Docker-aware)"""
    import os
    
    # Check if we're running in Docker
    database_url = os.getenv('DATABASE_URL')
    
    print(f"🔍 DATABASE_URL: {database_url}")  # Debug info
    
    if database_url:
        print("🐳 Running in Docker - using DATABASE_URL")
        try:
            # Parse DATABASE_URL: postgresql+asyncpg://postgres:password@database:5432/restaurant_recommendation
            if database_url.startswith('postgresql+asyncpg://'):
                database_url = database_url.replace('postgresql+asyncpg://', 'postgresql://')
            
            import urllib.parse
            result = urllib.parse.urlparse(database_url)
            
            connection_params = {
                'host': result.hostname,
                'database': result.path[1:],  # Remove leading /
                'user': result.username,
                'password': result.password,
                'port': result.port or 5432
            }
            
            print(f"🔗 Connecting to: {connection_params['host']}:{connection_params['port']}")
            
            return psycopg2.connect(**connection_params)
            
        except Exception as e:
            print(f"❌ Failed to parse DATABASE_URL: {e}")
            print("🔄 Falling back to hardcoded Docker settings...")
            
            # Fallback to hardcoded Docker database connection
            return psycopg2.connect(
                host="database",  # Docker service name
                database="restaurant_recommendation", 
                user="postgres",
                password="password",
                port="5432"
            )
    else:
        print("🖥️  Running locally - using localhost")
        # Running locally - use localhost
        return psycopg2.connect(
            host="localhost",
            database="restaurant_recommendation", 
            user="postgres",
            password="password",
            port="5432"
        )

def create_tables():
    """Create tables if they don't exist"""
    conn = connect_to_database()
    cur = conn.cursor()
    
    try:
        # Create users table with INTEGER user_id (not VARCHAR)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                f00 FLOAT NOT NULL, f01 FLOAT NOT NULL, f02 FLOAT NOT NULL, f03 FLOAT NOT NULL, f04 FLOAT NOT NULL,
                f05 FLOAT NOT NULL, f06 FLOAT NOT NULL, f07 FLOAT NOT NULL, f08 FLOAT NOT NULL, f09 FLOAT NOT NULL,
                f10 FLOAT NOT NULL, f11 FLOAT NOT NULL, f12 FLOAT NOT NULL, f13 FLOAT NOT NULL, f14 FLOAT NOT NULL,
                f15 FLOAT NOT NULL, f16 FLOAT NOT NULL, f17 FLOAT NOT NULL, f18 FLOAT NOT NULL, f19 FLOAT NOT NULL,
                f20 FLOAT NOT NULL, f21 FLOAT NOT NULL, f22 FLOAT NOT NULL, f23 FLOAT NOT NULL, f24 FLOAT NOT NULL,
                f25 FLOAT NOT NULL, f26 FLOAT NOT NULL, f27 FLOAT NOT NULL, f28 FLOAT NOT NULL, f29 FLOAT NOT NULL
            )
        """)
        
        # Create restaurants table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS restaurants (
                restaurant_id INTEGER PRIMARY KEY,
                latitude FLOAT NOT NULL,
                longitude FLOAT NOT NULL,
                f00 FLOAT NOT NULL, f01 FLOAT NOT NULL, f02 FLOAT NOT NULL, f03 FLOAT NOT NULL, f04 FLOAT NOT NULL,
                f05 FLOAT NOT NULL, f06 FLOAT NOT NULL, f07 FLOAT NOT NULL, f08 FLOAT NOT NULL, f09 FLOAT NOT NULL
            )
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_restaurant_id ON restaurants(restaurant_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants(latitude, longitude)")
        
        conn.commit()
        print("✅ Tables created successfully")
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def inspect_data_types():
    """Inspect the actual data types in parquet files"""
    print("🔍 Inspecting data types...")
    
    # Check user features
    user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
    print(f"   User features: {len(user_df):,} rows")
    print(f"   user_id type: {user_df['user_id'].dtype}")
    print(f"   user_id sample: {user_df['user_id'].head().tolist()}")
    
    # Check restaurant features
    restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    print(f"   Restaurant features: {len(restaurant_df):,} rows")
    print(f"   restaurant_id type: {restaurant_df['restaurant_id'].dtype}")
    print(f"   restaurant_id sample: {restaurant_df['restaurant_id'].head().tolist()}")
    
    # Check requests if it exists
    if (DATA_DIR / "requests.parquet").exists():
        requests_df = pd.read_parquet(DATA_DIR / "requests.parquet")
        print(f"   Requests: {len(requests_df):,} rows")
        print(f"   user_id type in requests: {requests_df['user_id'].dtype}")
        print(f"   user_id sample in requests: {requests_df['user_id'].head().tolist()}")

def load_user_features():
    """Load user features from parquet to PostgreSQL with proper type handling"""
    print("📊 Loading user features...")
    
    # Read user features
    user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
    print(f"   Found {len(user_df):,} users")
    print(f"   Original user_id type: {user_df['user_id'].dtype}")
    
    # Convert user_id to integer if it's not already
    if user_df['user_id'].dtype != 'int64':
        print("   Converting user_id to integer...")
        user_df['user_id'] = user_df['user_id'].astype('int64')
    
    conn = connect_to_database()
    cur = conn.cursor()
    
    try:
        # Clear existing data
        cur.execute("DELETE FROM users")
        print("   Cleared existing user data")
        
        # Insert users in batches
        batch_size = 1000
        total_batches = len(user_df) // batch_size + 1
        
        insert_sql = """
            INSERT INTO users (user_id, f00, f01, f02, f03, f04, f05, f06, f07, f08, f09,
                             f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, 
                             f20, f21, f22, f23, f24, f25, f26, f27, f28, f29)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for i in range(0, len(user_df), batch_size):
            batch = user_df.iloc[i:i+batch_size]
            
            # Convert batch to list of tuples with proper type conversion
            values = []
            for _, row in batch.iterrows():
                # Convert user_id to integer (not string!)
                user_id = int(row['user_id'])
                features = [float(val) if pd.notna(val) else 0.0 for val in row.values[1:]]  # Skip user_id
                user_tuple = (user_id,) + tuple(features)  # user_id as int
                values.append(user_tuple)
            
            cur.executemany(insert_sql, values)
            
            batch_num = i // batch_size + 1
            if batch_num % 100 == 0:
                print(f"   Users: {batch_num}/{total_batches} batches")
        
        conn.commit()
        print("✅ Users loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading users: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def load_restaurant_features():
    """Load restaurant features from parquet to PostgreSQL with proper coordinates"""
    print("🍽️  Loading restaurant features...")
    
    # Read restaurant features
    restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    print(f"   Found {len(restaurant_df):,} restaurants")
    print(f"   Original restaurant_id type: {restaurant_df['restaurant_id'].dtype}")
    
    # Check if latitude/longitude already exist
    has_coordinates = 'latitude' in restaurant_df.columns and 'longitude' in restaurant_df.columns
    print(f"   Has coordinates in file: {has_coordinates}")
    
    conn = connect_to_database()
    cur = conn.cursor()
    
    try:
        # Clear existing data
        cur.execute("DELETE FROM restaurants")
        print("   Cleared existing restaurant data")
        
        # Insert restaurants in batches
        batch_size = 1000
        total_batches = len(restaurant_df) // batch_size + 1
        
        insert_sql = """
            INSERT INTO restaurants (restaurant_id, latitude, longitude, f00, f01, f02, f03, f04, f05, f06, f07, f08, f09)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for i in range(0, len(restaurant_df), batch_size):
            batch = restaurant_df.iloc[i:i+batch_size]
            
            # Prepare data with coordinates
            values = []
            for _, row in batch.iterrows():
                restaurant_id = int(row['restaurant_id'])
                
                # Use existing coordinates or generate random ones
                if has_coordinates:
                    lat = float(row['latitude'])
                    lon = float(row['longitude'])
                else:
                    # Generate random coordinates (Bangkok area)
                    lat = float(np.random.uniform(13.5, 14.0))
                    lon = float(np.random.uniform(100.3, 100.9))
                
                # Get feature columns (f00-f09)
                feature_cols = [f'f{i:02d}' for i in range(10)]
                features = [float(row[col]) for col in feature_cols]
                
                # Create tuple: restaurant_id, lat, lon, f00-f09
                value = (restaurant_id, lat, lon, *features)
                values.append(value)
            
            cur.executemany(insert_sql, values)
            
            batch_num = i // batch_size + 1
            if batch_num % 100 == 0:
                print(f"   Restaurants: {batch_num}/{total_batches} batches")
        
        conn.commit()
        print("✅ Restaurants loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading restaurants: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def verify_data():
    """Verify loaded data with proper type checking"""
    print("🔍 Verifying data...")
    
    conn = connect_to_database()
    cur = conn.cursor()
    
    try:
        # Count users
        cur.execute("SELECT COUNT(*) FROM users")
        user_count = cur.fetchone()[0]
        
        # Count restaurants
        cur.execute("SELECT COUNT(*) FROM restaurants") 
        restaurant_count = cur.fetchone()[0]
        
        # Sample user (check type)
        cur.execute("SELECT user_id, pg_typeof(user_id) FROM users LIMIT 1")
        sample_user = cur.fetchone()
        
        # Sample restaurant (check type)
        cur.execute("SELECT restaurant_id, pg_typeof(restaurant_id) FROM restaurants LIMIT 1")
        sample_restaurant = cur.fetchone()
        
        # Check user_id range
        cur.execute("SELECT MIN(user_id), MAX(user_id) FROM users")
        user_range = cur.fetchone()
        
        # Check restaurant_id range
        cur.execute("SELECT MIN(restaurant_id), MAX(restaurant_id) FROM restaurants")
        restaurant_range = cur.fetchone()
        
        print(f"✅ Database verification:")
        print(f"   Users in DB: {user_count:,}")
        print(f"   Restaurants in DB: {restaurant_count:,}")
        print(f"   Sample user: {sample_user[0]} (type: {sample_user[1]})")
        print(f"   Sample restaurant: {sample_restaurant[0]} (type: {sample_restaurant[1]})")
        print(f"   User ID range: {user_range[0]} - {user_range[1]}")
        print(f"   Restaurant ID range: {restaurant_range[0]} - {restaurant_range[1]}")
        
        # Verify no duplicate IDs
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM users")
        unique_users = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT restaurant_id) FROM restaurants")
        unique_restaurants = cur.fetchone()[0]
        
        print(f"   Unique users: {unique_users:,} (should equal total)")
        print(f"   Unique restaurants: {unique_restaurants:,} (should equal total)")
        
        if unique_users != user_count:
            print("   ⚠️  Warning: Duplicate user IDs found!")
        
        if unique_restaurants != restaurant_count:
            print("   ⚠️  Warning: Duplicate restaurant IDs found!")
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to load all data with type inspection"""
    try:
        print("🔄 Starting data loading process...")
        
        # First, inspect the data types
        inspect_data_types()
        
        # Create tables
        create_tables()
        
        # Load data
        load_user_features()
        load_restaurant_features()
        verify_data()
        
        print("🎉 Data loading completed successfully!")

        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()