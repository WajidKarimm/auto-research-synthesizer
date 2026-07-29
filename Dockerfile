FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System packages needed for some Python packages (psycopg2, build tools)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libpq-dev \
       ffmpeg \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY . /app

# Expose ports for API (uvicorn) and optional Streamlit UI
EXPOSE 8000 8501

# Default command: start the FastAPI app. Override to run Streamlit if desired.
CMD ["uvicorn", "ars.main:app", "--host", "0.0.0.0", "--port", "8000"]
