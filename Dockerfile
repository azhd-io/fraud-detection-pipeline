# Use a small, official Python image
FROM python:3.11-slim

# 1) System deps (needed by numpy/scipy/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Keep Python output unbuffered & no .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3) Set workdir
WORKDIR /app

# 4) Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5) Copy project code (and models) into the image
COPY . .

# 6) Expose API port
EXPOSE 8000

# 7) Start the FastAPI server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
