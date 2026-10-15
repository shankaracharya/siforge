FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential procps && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/siforge
COPY pyproject.toml README.md ./
COPY siforge ./siforge
RUN pip install --no-cache-dir ".[structure,parquet]"
WORKDIR /work
ENTRYPOINT ["siforge"]
CMD ["--help"]
