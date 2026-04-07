# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Set work directory
WORKDIR /app

# Install minimal system libraries for OpenCV/Imaging
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Ensure the upload directory exists and is writable
RUN mkdir -p static/uploads && chmod 777 static/uploads

# Expose the port (7860 by default for HF Spaces)
EXPOSE 7860

# Run the application using Gunicorn for production
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 2 app:app"]
