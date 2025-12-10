FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install .

COPY backend ./backend

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0","--reload", "--port", "8000"]
