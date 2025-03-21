# syntax=docker/dockerfile:1

# Use the official Python image as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY vinylitics vinylitics

# Create a non-root user and switch to it for security
RUN useradd -m appuser
USER appuser

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application when the container launches
CMD ["uvicorn", "vinylitics.api.fast:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
