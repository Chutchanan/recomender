#!/bin/bash
# start.sh - Easy startup script for Restaurant Recommendation API

set -e

echo "🚀 Starting Restaurant Recommendation API..."

# Check if data/model.pt exists
if [ ! -f "data/model.pt" ]; then
    echo "❌ Model file not found: data/model.pt"
    echo "Please ensure you have trained your model first using:"
    echo "python 02_train_model_optimized.py"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "📊 Starting database..."
docker-compose up -d database

echo "⏳ Waiting for database to be ready..."
sleep 10

# Check if we need to load data
echo "🔍 Checking if database has data..."
USER_COUNT=$(docker-compose exec -T database psql -U postgres -d restaurant_recommendation -t -c "SELECT COUNT(*) FROM users;" 2>/dev/null || echo "0")

if [ "$USER_COUNT" -eq "0" ] 2>/dev/null; then
    echo "📥 Database is empty. Loading data..."
    docker-compose --profile setup up data-loader
    echo "✅ Data loaded successfully"
else
    echo "✅ Database already has data (${USER_COUNT} users)"
fi

echo "🚀 Starting API server..."
docker-compose up -d api

echo "⏳ Waiting for API to start..."
sleep 15

# Check if API is healthy
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy and ready!"
    echo ""
    echo "🌐 API Endpoints:"
    echo "   • Health check: http://localhost:8000/health"
    echo "   • Documentation: http://localhost:8000/docs"
    echo "   • Simple prediction: POST http://localhost:8000/predict/{user_id}"
    echo "   • Full recommendation: POST http://localhost:8000/recommend/{user_id}"
    echo "   • Files prediction: POST http://localhost:8000/predict-files/{user_id}"
    echo ""
    echo "📝 Test the API:"
    echo 'curl -X POST "http://localhost:8000/predict/0" -H "Content-Type: application/json" -d '"'"'{"candidate_restaurant_ids": [1,2,3,4,5]}'"'"''
    echo ""
    echo "🔧 View logs: docker-compose logs -f api"
    echo "🛑 Stop services: docker-compose down"
else
    echo "❌ API failed to start. Check logs:"
    docker-compose logs api
    exit 1
fi