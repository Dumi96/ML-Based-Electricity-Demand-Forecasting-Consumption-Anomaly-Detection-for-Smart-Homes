FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

RUN pip install --upgrade pip && 
    pip install -r requirements.txt

COPY api ./api
COPY app ./app
COPY src ./src
COPY models ./models

EXPOSE 8000 8501
