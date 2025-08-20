# Restaurant Recommendation API v3.0

A simplified, high-performance restaurant recommendation service using PostgreSQL + PyTorch model.

## 🏗️ Project Structure

```
restaurant-recommendation-api/
├── adapters/
│   ├── api.py              # FastAPI controller (single endpoint)
│   └── database.py         # PostgreSQL adapter + Domain models
├── services/
│   ├── __init__.py         # Empty init file
│   └── recommend_model.py  # Model service (PostgreSQL + model.pt)
├── scripts/
│   ├── init_db.sql         # Database schema
│   └── load_data.py        # Data loader
├── data/                   # Your data files
│   ├── model.pt            # Trained PyTorch model (required)
│   ├── user_features.parquet
│   └── restaurant_features.parquet
├── config.py               # Configuration
├── main.py                 # Application entry point
├── requirements.txt        # Dependencies
├── Dockerfile              # Container setup
├── docker-compose.yml      # Services orchestration
├── start.sh / start.bat    # Easy startup scripts
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Your trained model file: `data/model.pt`
- Your data files: `data/user_features.parquet`, `data/restaurant_features.parquet`

### Windows Users


#### Command Prompt
```cmd
start.bat
```


### Linux/Mac Users

#### Start services
```bash
chmod +x start.sh
./start.sh
```

## 📚 API Endpoints

### Main Prediction Endpoint
```bash
curl -X POST "http://localhost:8000/predict/0" \
  -H "Content-Type: application/json" \
  -d '{"candidate_restaurant_ids": [1,2,3,4,5]}'
```

**Response:**
```json
{
  "user_id": "0",
  "recommendations": [
    {
      "restaurant_id": 3,
      "score": 0.8567,
      "latitude": 13.7563,
      "longitude": 100.5018
    },
    {
      "restaurant_id": 1,
      "score": 0.7234,
      "latitude": 13.7465,
      "longitude": 100.5120
    }
  ],
  "total_candidates": 5,
  "processing_time_ms": 45.2
}
```

### Other Endpoints
```bash
# Health Check
curl http://localhost:8000/health

# Model Information
curl http://localhost:8000/model/info

# API Documentation (Swagger UI)
open http://localhost:8000/docs
```

## 🏛️ Architecture

```
API → RecommendModelService → PostgreSQL + model.pt
```

**Simple Flow:**
1. **Request**: User ID + candidate restaurant IDs
2. **Database**: Get user features (30) + restaurant features (10 each) from PostgreSQL
3. **Model**: Combine features → Predict with model.pt → Get click probabilities
4. **Response**: Restaurants sorted by score (highest first)

**Key Features:**
- ✅ **Single purpose**: PostgreSQL data + model.pt predictions only
- ✅ **Clean separation**: Services layer for model logic
- ✅ **Domain models**: User/Restaurant entities in database adapter
- ✅ **Simple API**: One main prediction endpoint
- ✅ **Fast predictions**: Batch processing for multiple restaurants

## 🔧 Manual Setup (without Docker)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup PostgreSQL
```bash
# Start PostgreSQL (adjust for your system)
sudo systemctl start postgresql

# Create database
createdb restaurant_recommendation
psql restaurant_recommendation < scripts/init_db.sql
```

### 3. Load data
```bash
python scripts/load_data.py
```

### 4. Start API
```bash
python main.py
```


## 🛠️ Development

### View logs
```bash
docker-compose logs -f api
```

### Stop services
```bash
docker-compose down
```

### Rebuild
```bash
docker-compose build
docker-compose up -d
```

## 📊 Performance

- **Throughput**: ~1000 RPS
- **Latency**: <100ms for typical requests
- **Batch processing**: Optimized for multiple restaurant predictions per user
- **Memory efficient**: Single model load, reused for all predictions
- **Database pooling**: PostgreSQL connection pool for concurrent requests

## 🔄 Changes from Previous Versions

### **v3.0 (Current) - Services Architecture**
- ✅ **Model service**: Separated into `services/recommend_model.py`
- ✅ **Single endpoint**: One main `/predict/{user_id}` endpoint
- ✅ **PostgreSQL focus**: Only uses database data (no parquet file options)
- ✅ **Domain models**: Moved to database adapter layer
- ✅ **Simplified**: Removed complex business logic layer


## 💡 When to Use This API

This API is perfect when you:
- ✅ Have user and restaurant data in PostgreSQL
- ✅ Want simple, fast predictions using a trained model
- ✅ Need to score multiple restaurants for one user
- ✅ Prefer a clean, maintainable architecture
- ✅ Want to deploy with Docker
