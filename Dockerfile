FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install prettier globally using npm
RUN npm install -g prettier@3.4.2
RUN pip install uv

# Copy requirements first to leverage Docker cache

# Copy the rest of the application
COPY . .

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV DATA_DIR=/app/data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "main.py"]