FROM python:3.11-slim

# opencv-python-headless needs libGL and glib at runtime even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker layer-caches them across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app.
COPY . .

# Render provides $PORT at runtime; app.py reads it and binds 0.0.0.0.
# Default to 7860 for local `docker run` without -e PORT.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
