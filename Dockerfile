FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home repro
WORKDIR /repro

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=repro:repro . .
USER repro
ENTRYPOINT ["./run.sh"]
CMD ["all"]
