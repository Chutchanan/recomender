# load_data_simple.py - Simple PostgreSQL data loader using psycopg2
import pandas as pd
import psycopg2
from pathlib import Path
import time
import numpy as np

DATA_DIR = Path("data")

def connect_to_database():
    """Connect to PostgreSQL database"""
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
        # Create users table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
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

def load_user_features():
    """Load user features from parquet to PostgreSQL"""
    print("📊 Loading user features...")
    
    # Read user features
    user_df = pd.read_parquet(DATA_DIR / "user_features.parquet")
    print(f"   Found {len(user_df):,} users")
    
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
                # Keep user_id as string, convert features to float
                user_id = str(row['user_id'])  # Keep as string
                features = [float(val) if pd.notna(val) else 0.0 for val in row.values[1:]]  # Skip user_id
                user_tuple = (user_id,) + tuple(features)
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
    """Load restaurant features from parquet to PostgreSQL"""
    print("🍽️  Loading restaurant features...")
    
    # Read restaurant features
    restaurant_df = pd.read_parquet(DATA_DIR / "restaurant_features.parquet")
    print(f"   Found {len(restaurant_df):,} restaurants")
    
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
            
            # Prepare data with random coordinates and proper type conversion
            values = []
            for _, row in batch.iterrows():
                # Generate random coordinates (Bangkok area)
                lat = float(np.random.uniform(13.5, 14.0))
                lon = float(np.random.uniform(100.3, 100.9))
                
                # Create tuple: restaurant_id, lat, lon, f00-f09 (convert all to native Python types)
                value = (
                    int(row['restaurant_id']), lat, lon,
                    float(row['f00']), float(row['f01']), float(row['f02']), 
                    float(row['f03']), float(row['f04']), float(row['f05']), 
                    float(row['f06']), float(row['f07']), float(row['f08']), float(row['f09'])
                )
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
    """Verify loaded data"""
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
        
        # Sample user
        cur.execute("SELECT user_id FROM users LIMIT 1")
        sample_user = cur.fetchone()
        
        # Sample restaurant
        cur.execute("SELECT restaurant_id FROM restaurants LIMIT 1")
        sample_restaurant = cur.fetchone()
        
        print(f"✅ Database verification:")
        print(f"   Users in DB: {user_count:,}")
        print(f"   Restaurants in DB: {restaurant_count:,}")
        print(f"   Sample user: {sample_user[0] if sample_user else 'None'}")
        print(f"   Sample restaurant ID: {sample_restaurant[0] if sample_restaurant else 'None'}")
        
    except Exception as e:
        print(f"❌ Error verifying data: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    """Main function to load all data"""
    try:
        print("🔄 Starting data loading process...")
        
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