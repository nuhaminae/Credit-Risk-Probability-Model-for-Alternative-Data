# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy everything in your project into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "scripts.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
