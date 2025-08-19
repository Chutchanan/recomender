# fix_user_ids.py - Fix user IDs to proper string format
import psycopg2

def fix_user_ids():
    """Convert user IDs from float format to clean string format"""
    print("🔧 Fixing user IDs...")
    
    # Connect to database
    conn = psycopg2.connect(
        host="localhost",
        database="restaurant_recommendation",
        user="postgres",
        password="password"
    )
    cur = conn.cursor()
    
    try:
        # Get all current user IDs
        cur.execute("SELECT user_id FROM users")
        user_ids = cur.fetchall()
        print(f"Found {len(user_ids)} users to fix")
        
        # Update each user ID to clean format
        for i, (old_user_id,) in enumerate(user_ids):
            # Convert "0.0" -> "0", "1.0" -> "1", etc.
            new_user_id = str(int(float(old_user_id)))
            
            cur.execute(
                "UPDATE users SET user_id = %s WHERE user_id = %s",
                (new_user_id, old_user_id)
            )
            
            if (i + 1) % 10000 == 0:
                print(f"   Fixed {i + 1}/{len(user_ids)} user IDs")
        
        conn.commit()
        
        # Verify the fix
        cur.execute("SELECT user_id FROM users LIMIT 5")
        sample_ids = cur.fetchall()
        
        print("✅ User IDs fixed successfully!")
        print("Sample user IDs after fix:")
        for user_id, in sample_ids:
            print(f"   {user_id}")
            
    except Exception as e:
        print(f"❌ Error fixing user IDs: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    fix_user_ids()