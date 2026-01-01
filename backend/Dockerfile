FROM python:3.11-slim

WORKDIR /app

# Install dependencies (Layer Caching)
COPY pyproject.toml .
# Fake src to satisfy pip install .
RUN mkdir src && touch src/__init__.py && touch README.md
RUN pip install .

# Copy actual code
COPY . .
# Re-install to link the package
RUN pip install .

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]