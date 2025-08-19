@echo off
REM start_services.bat - Windows startup script for Restaurant Recommendation System

echo 🚀 Starting Restaurant Recommendation System
echo ============================================

REM Step 1: Start database
echo 📊 Starting PostgreSQL database...
docker-compose up -d database

REM Wait for database to be healthy
echo ⏳ Waiting for database to be ready...
:wait_db
docker-compose ps database | findstr "healthy" >nul
if errorlevel 1 (
    timeout /t 2 >nul
    goto wait_db
)
echo ✅ Database is healthy!

REM Step 2: Check if data exists
echo 🔍 Checking if data exists...
for /f %%i in ('docker exec restaurant_db psql -U postgres -d restaurant_recommendation -t -c "SELECT COUNT(*) FROM users;" 2^>nul') do set user_count=%%i

if "%user_count%"=="0" (
    echo 📥 Loading initial data...
    docker-compose --profile setup up data-loader
    if errorlevel 1 (
        echo ❌ Data loading failed
        pause
        exit /b 1
    )
    echo ✅ Data loaded successfully
) else (
    echo ✅ Data already exists ^(%user_count% users found^)
)

REM Step 3: Start API service
echo 🚀 Starting API service...
docker-compose up -d api

REM Wait for API to be healthy
echo ⏳ Waiting for API to be ready...
:wait_api
docker-compose ps api | findstr "healthy" >nul
if errorlevel 1 (
    timeout /t 3 >nul
    goto wait_api
)
echo ✅ API is healthy!

REM Step 4: Show status and URLs
echo.
echo 🎉 All services are running!
echo ============================================
echo 📊 Database:     http://localhost:5432
echo 🌐 API:          http://localhost:8000
echo 📚 API Docs:     http://localhost:8000/docs
echo ❤️  Health:      http://localhost:8000/health
echo.
echo 📋 Quick test commands:
echo   curl http://localhost:8000/health
echo   curl http://localhost:8000/model/info
echo.
echo 🔧 Management commands:
echo   docker-compose logs api      ^# View API logs
echo   docker-compose logs database ^# View DB logs
echo   docker-compose down          ^# Stop all services
echo.

REM Optional: Run a quick health check
echo 🔧 Running health check...
timeout /t 5 >nul

curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Health check failed - API may still be starting up
) else (
    echo ✅ Health check passed!
)

echo 🚀 Setup complete! Services are ready.
echo Press any key to continue...
pause >nul