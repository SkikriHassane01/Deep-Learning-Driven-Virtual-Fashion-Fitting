# Virtual Fashion Try-On Dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DJANGO_SETTINGS_MODULE=virtual_tryon.settings.production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    libgeos++-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/webapp/media/uploads && \
    mkdir -p /app/webapp/media/results && \
    mkdir -p /app/webapp/staticfiles && \
    mkdir -p /app/models/checkpoints

# Set permissions
RUN chmod -R 755 /app

# Change to webapp directory
WORKDIR /app/webapp

# Collect static files and run migrations
RUN python manage.py collectstatic --noinput --settings=virtual_tryon.settings.production
RUN python manage.py migrate --settings=virtual_tryon.settings.production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/', timeout=10)" || exit 1

# Run the application
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000", "--settings=virtual_tryon.settings.production"]