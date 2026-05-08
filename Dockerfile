# 1. Use the official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level library required by XGBoost C++ backend
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY requirements first (Smart caching, just like you did before)
COPY requirements.txt .

# 5. Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your app and pickle files
COPY . .

# 7. Expose the port
EXPOSE 10000

# 8. The Ignition Switch for FastAPI
# Make sure your Python file is named 'app.py' or change this to 'main:app'
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]