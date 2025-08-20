@echo off
REM start.bat - Windows startup script for Restaurant Recommendation API

echo 🚀 Starting Restaurant Recommendation API...

REM Check if data/model.pt exists
if not exist "data\model.pt" (
    echo ❌ Model file not found: data\model.pt
    echo Please ensure you have trained your model first using:
    echo python 02_train_model_optimized.py
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo 📊 Starting database...
docker-compose up -d database

echo ⏳ Waiting for database to be ready...
timeout /t 10 /nobreak >nul

REM Check if we need to load data
echo 🔍 Checking if database has data...
for /f %%i in ('docker-compose exec -T database psql -U postgres -d restaurant_recommendation -t -c "SELECT COUNT(*) FROM users;" 2^>nul ^| findstr /r "[0-9]"') do set USER_COUNT=%%i

if "%USER_COUNT%"=="" set USER_COUNT=0

if %USER_COUNT% equ 0 (
    echo 📥 Database is empty. Loading data...
    docker-compose --profile setup up data-loader
    echo ✅ Data loaded successfully
) else (
    echo ✅ Database already has data ^(%USER_COUNT% users^)
)

echo 🚀 Starting API server...
docker-compose up -d api

echo ⏳ Waiting for API to start...
timeout /t 15 /nobreak >nul

REM Check if API is healthy
curl -f http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API failed to start. Check logs:
    docker-compose logs api
    pause
    exit /b 1
)

echo ✅ API is healthy and ready!
echo.
echo 🌐 API Endpoints:
echo    • Health check: http://localhost:8000/health
echo    • Documentation: http://localhost:8000/docs
echo    • Simple prediction: POST http://localhost:8000/predict/{user_id}
echo    • Full recommendation: POST http://localhost:8000/recommend/{user_id}
echo    • Files prediction: POST http://localhost:8000/predict-files/{user_id}
echo.
echo 📝 Test the API:
echo curl -X POST "http://localhost:8000/predict/0" -H "Content-Type: application/json" -d "{\"candidate_restaurant_ids\": [1,2,3,4,5]}"
echo.
echo 🔧 View logs: docker-compose logs -f api
echo 🛑 Stop services: docker-compose down
echo.
pause