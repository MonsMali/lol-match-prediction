# ------------------------------------------------------------------
# Hugging Face Spaces Docker deployment
# Serves the FastAPI backend + compiled React SPA on port 7860
# ------------------------------------------------------------------

# Stage 1: Build the React frontend
FROM node:22-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Production image
FROM python:3.12-slim
WORKDIR /app

# Install production Python dependencies
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Run the server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
