# Dockerfile
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and trained model into the container
COPY src/ src/
COPY models/ models/

# Expose API port
EXPOSE 8080

# Start the app with gunicorn (production server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "src.app:app"]
