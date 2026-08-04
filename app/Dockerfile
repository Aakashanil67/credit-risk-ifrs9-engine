# Streamlit dashboard. Same models/ handling as api/Dockerfile — see that file's comment.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY models/ models/

EXPOSE 8501

# API_URL is overridden by docker-compose.yml to point at the api service by its compose name —
# "localhost" would mean "this container" inside Docker's network, not the sibling api container.
ENV API_URL=http://localhost:8000

CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
