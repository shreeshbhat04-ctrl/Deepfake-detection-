# 1. Use Python 3.11-slim to keep the image size small and stable
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for OpenCV
# These are linux libraries that opencv-python-headless needs to run
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install Python dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Expose port 8080 (Standard for Cloud Run)
ENV PORT=8080
EXPOSE 8080

# 8. Command to run the API using Uvicorn
# Host 0.0.0.0 is required for container networking
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]