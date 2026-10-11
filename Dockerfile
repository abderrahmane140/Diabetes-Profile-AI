FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY requirements-app.txt .

RUN pip install --no-cache-dir -r requirements-app.txt

COPY src ./src
COPY api ./api
COPY frontend ./frontend