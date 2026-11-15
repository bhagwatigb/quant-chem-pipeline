FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline script
COPY pipeline.py .

# Create mount point directories
RUN mkdir -p /app/quant_chem/notes /app/quant_chem/assets /app/quant_chem/transcription

ENTRYPOINT ["python", "pipeline.py"]
CMD ["--help"]

