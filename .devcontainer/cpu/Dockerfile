ARG VARIANT

FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9-bullseye

# Requires $POETRY_HOME to install the location; poetry is installed under /root by default.
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/vscode/.local python3 -
