
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app/app
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODELS=ministral-3:3b,qwen3-vl:4b,gemma3:4b-it-qat,llava:latest
ENV OLLAMA_MODEL=gemma3:4b-it-qat
ENV DATA_DIR=/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
