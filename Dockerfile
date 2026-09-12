# Reproducible host for ANBA experiments. Traces are downloaded at runtime
# (not copied into the image). ChampSim is cloned at the pinned commit.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
        xz-utils \
        zip \
        unzip \
        pkg-config \
        ninja-build \
        tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/anba
COPY requirements.txt /opt/anba/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

COPY . /opt/anba

# ChampSim + traces are fetched by Makefile targets, not baked in.
CMD ["make", "help"]
