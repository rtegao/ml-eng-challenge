# Use the official Python 3.10 image as base
FROM python:3.10

# Install dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.1.14

# Set environment variables
ENV PATH="$PATH:/root/.poetry/bin"
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/train_pipeline

# Copy only the dependency files to optimize caching
WORKDIR /train_pipeline
COPY poetry.lock pyproject.toml /train_pipeline/

# Install project dependencies
RUN poetry install --no-root --no-dev

# Copy the rest of the application code
COPY . .