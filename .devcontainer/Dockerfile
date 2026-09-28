ARG VARIANT

FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Requires $POETRY_HOME to install the location; poetry is installed under /root by default.
# Pin the version to match poetry.lock; newer Poetry no longer supports Python 3.9.
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/vscode/.local POETRY_VERSION=1.8.4 python3 -
