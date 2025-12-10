FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .

RUN pip install .

COPY frontend ./frontend

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

WORKDIR /app/frontend

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
