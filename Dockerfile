# Dockerfile - Multi-stage build for the complete application
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage for API serving
FROM base as production

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run API server
CMD ["python", "main.py"]

# Development stage (for local development)
FROM base as development
COPY . .
CMD ["python", "main.py"]

# Data loading stage (for initial data setup)
FROM base as data-loader
COPY . .
CMD ["python", "load_data_simple.py"]