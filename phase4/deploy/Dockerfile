# Build from the REPOSITORY ROOT so phase1-3 data and models are inside the image (they are read-only inputs):
#   docker build -f phase4/deploy/Dockerfile -t cortexhr-api .
#   docker run -p 8000:8000 -e CORTEXHR_ENV=production cortexhr-api
FROM python:3.12-slim
WORKDIR /app
COPY phase4/requirements.txt phase4/requirements.txt
RUN pip install --no-cache-dir -r phase4/requirements.txt
COPY phase1 phase1
COPY phase2 phase2
COPY phase3 phase3
COPY phase4 phase4
ENV PYTHONPATH=/app/phase4/src PYTHONUNBUFFERED=1 CORTEXHR_ENV=production
EXPOSE 8000
CMD ["sh", "-c", "uvicorn cortexhr_phase4.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
