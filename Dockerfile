FROM python:3.10-slim

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir fastapi uvicorn scikit-learn

# Expose port for the API
EXPOSE 8000

# Default command to start the service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]