FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for scikit-learn, nltk, etc.
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

# Copy the rest of the project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the app using Gunicorn (already in your requirements.txt)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]