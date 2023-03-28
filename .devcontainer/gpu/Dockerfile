FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    # Remove imagemagick due to https://security-tracker.debian.org/tracker/CVE-2019-10131, which is done in Microsoft's devcontainer image.
    && apt-get purge -y imagemagick imagemagick-6-common \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    curl \
    libboost-dev \
    libpython3-dev \
    manpages \
    man-db \
    python3 \
    python3-distutils \
    python3-pip \
    wget \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/* \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/vscode/.local python3 -
