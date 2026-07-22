FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for ultralytics (YOLO)
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080

CMD ["python", "run.py"]
