-- scripts/init_db.sql - Database initialization (matching your schema)

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(255) PRIMARY KEY,
    f00 FLOAT NOT NULL,
    f01 FLOAT NOT NULL,
    f02 FLOAT NOT NULL,
    f03 FLOAT NOT NULL,
    f04 FLOAT NOT NULL,
    f05 FLOAT NOT NULL,
    f06 FLOAT NOT NULL,
    f07 FLOAT NOT NULL,
    f08 FLOAT NOT NULL,
    f09 FLOAT NOT NULL,
    f10 FLOAT NOT NULL,
    f11 FLOAT NOT NULL,
    f12 FLOAT NOT NULL,
    f13 FLOAT NOT NULL,
    f14 FLOAT NOT NULL,
    f15 FLOAT NOT NULL,
    f16 FLOAT NOT NULL,
    f17 FLOAT NOT NULL,
    f18 FLOAT NOT NULL,
    f19 FLOAT NOT NULL,
    f20 FLOAT NOT NULL,
    f21 FLOAT NOT NULL,
    f22 FLOAT NOT NULL,
    f23 FLOAT NOT NULL,
    f24 FLOAT NOT NULL,
    f25 FLOAT NOT NULL,
    f26 FLOAT NOT NULL,
    f27 FLOAT NOT NULL,
    f28 FLOAT NOT NULL,
    f29 FLOAT NOT NULL
);

-- Create restaurants table
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id INTEGER PRIMARY KEY,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    f00 FLOAT NOT NULL,
    f01 FLOAT NOT NULL,
    f02 FLOAT NOT NULL,
    f03 FLOAT NOT NULL,
    f04 FLOAT NOT NULL,
    f05 FLOAT NOT NULL,
    f06 FLOAT NOT NULL,
    f07 FLOAT NOT NULL,
    f08 FLOAT NOT NULL,
    f09 FLOAT NOT NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_restaurants_restaurant_id ON restaurants(restaurant_id);
CREATE INDEX IF NOT EXISTS idx_restaurants_location ON restaurants(latitude, longitude);

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE users TO postgres;
GRANT ALL PRIVILEGES ON TABLE restaurants TO postgres;