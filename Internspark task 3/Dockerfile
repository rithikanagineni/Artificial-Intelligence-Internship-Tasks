# 1. Start from an official, optimized lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install basic system utilities to keep things clean
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file into the container
COPY requirements.txt .

# 5. Install Python dependencies (no-cache speeds up the build process)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy code files and model weights into the container
COPY app.py .
COPY binary_image_classifier.pth .

# 7. Expose the port FastAPI will run on
EXPOSE 8000

# 8. Command to launch the API using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]