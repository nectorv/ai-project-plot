FROM python:3.11-slim

WORKDIR /app

# Install uv (provides uvx, used by mcp_pool.py)
RUN pip install --no-cache-dir uv

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
