FROM  python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY params.yaml ./params.yaml
COPY data/models/model.ts.pt ./data/models/model.ts.pt

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.app.api:app", "--host", "0.0.0.0", "--port", "8000"]