# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH for GitHub
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN --mount=type=ssh pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY vinylitics vinylitics

# Create a non-root user and switch to it
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "vinylitics.api.fast:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
