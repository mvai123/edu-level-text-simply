FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    zstd \
  && rm -rf /var/lib/apt/lists/*

# Install Ollama (for local llama3 backend)
RUN curl -fsSL https://ollama.com/install.sh | sh

RUN pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml /app/pyproject.toml
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --project /app --all-groups --no-dev
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Code is expected to be bind-mounted into /app at runtime.
# Default to showing CLI help
ENTRYPOINT ["sh", "-lc", "sh /app/entrypoint.sh \"$@\"", "--"]
CMD ["python", "-m", "baselines.wrapper.cli", "--help"]
