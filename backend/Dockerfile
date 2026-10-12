FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/rion-yolo \
    MPLCONFIGDIR=/tmp/rion-matplotlib

WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt

COPY backend/app /app/app
COPY model/ /app/model/
COPY scripts/seed_policy_db.py /app/scripts/seed_policy_db.py
COPY infra/aws/db/schema.sql /app/infra/aws/db/schema.sql
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
