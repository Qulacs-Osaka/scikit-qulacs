# Pick a Python version: 3, 3.10, 3.9, 3.8, 3.7, 3.6
# Append -bullseye or -buster to pin to an OS version.
# Use -bullseye variants on local on arm64/Apple Silicon.
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9-bullseye

# Requires $POETRY_HOME to install the location; poetry is installed under /root by default.
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/vscode/.local python3 -
