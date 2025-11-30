# 1. Use 'bookworm' (Debian 12) - it is newer and fixes the "apt-get update" error
FROM python:3.11-slim-bookworm

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies
# We added 'build-essential' and 'cmake' to ensure dlib/mtcnn can build
# We switched 'libgl1-mesa-glx' to 'libgl1' because the old one is deprecated
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip
RUN pip install --upgrade pip

# 5. Copy requirements
COPY requirements.txt .

# 6. Install Python dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the application code
COPY . .

CMD uvicorn api:app --host 0.0.0.0 --port $PORT
