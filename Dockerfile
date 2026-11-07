FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.lock ./
COPY pixeldebug ./pixeldebug
RUN pip install --no-cache-dir -r requirements.lock && pip install --no-cache-dir --no-deps . && useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "pixeldebug.api:app", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/api"]
