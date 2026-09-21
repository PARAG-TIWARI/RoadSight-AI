# RoadSight AI — Deployment & Operations Guide

## 1. Architecture Overview for Deployment

RoadSight AI is designed as two decoupled services:
1. **Backend**: Python 3.11+ FastAPI microservice serving REST endpoints with ONNX Runtime inference engine.
2. **Frontend**: Single Page Application (SPA) built with React 18, Vite, and Tailwind CSS.

---

## 2. Local Development Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Node.js 18+ and npm 9+
- Git

### Backend Setup
```bash
# 1. Navigate to backend directory
cd backend

# 2. (Optional) Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp ../.env.example .env

# 5. Start Uvicorn development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Backend API will be accessible at `http://localhost:8000` with Swagger UI at `http://localhost:8000/docs`.

### Frontend Setup
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start Vite dev server
npm run dev
```
Frontend interface will be accessible at `http://localhost:5173`.

---

## 3. Containerized Deployment (Docker & Compose)

### Single-Command Deployment via Docker Compose
RoadSight AI includes a production-ready `docker-compose.yml` for unified local or server deployment:

```bash
# Build and run containers in detached mode
docker-compose up -d --build

# Inspect logs
docker-compose logs -f

# Stop containers
docker-compose down
```

This launches:
- **Backend container**: `http://localhost:8000`
- **Frontend container**: `http://localhost:80` (or configured port)

### Standalone Backend Docker
```bash
cd backend
docker build -t roadsight-backend:latest .
docker run -p 8000:8000 --env-file ../.env roadsight-backend:latest
```

### Standalone Frontend Docker
```bash
cd frontend
docker build -t roadsight-frontend:latest .
docker run -p 80:80 roadsight-frontend:latest
```

---

## 4. Production Cloud Deployment

### 4.1 Backend Deployment (Render / Railway / Fly.io)

> [!IMPORTANT]
> **Free Tier RAM Constraint**:
> Free hosting tiers (such as Render's 512MB RAM tier) will fail if loading PyTorch/Ultralytics due to memory exhaustion. RoadSight AI solves this with its **ONNX Runtime engine**, requiring only ~180MB RAM.
> Ensure `MODEL_FORMAT=onnx` is set in your production environment variables.

#### Deploying to Render:
1. Create a new **Web Service** on Render connected to your repository.
2. Configure settings:
   - **Root Directory**: `backend`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2`
3. Add Environment Variables:
   - `MODEL_FORMAT`: `onnx`
   - `MODEL_PATH`: `best.onnx`
   - `CORS_ORIGINS`: `https://your-frontend.vercel.app,http://localhost:5173`
   - `MAX_VIDEO_FRAMES`: `300`
   - `CONFIDENCE_THRESHOLD`: `0.4`

### 4.2 Frontend Deployment (Vercel / Netlify)

#### Deploying to Vercel:
1. Import repository on Vercel.
2. Configure project:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
3. Add Environment Variable:
   - `VITE_API_URL`: `https://your-backend.onrender.com`
4. Deploy!

---

## 5. Health Checks & Verification Checklist

1. **Verify Backend Health**:
   ```bash
   curl -s http://localhost:8000/api/v1/health | jq .
   ```
2. **Verify Model Metadata**:
   ```bash
   curl -s http://localhost:8000/api/v1/model | jq .
   ```
3. **Run In-Service Benchmark**:
   ```bash
   curl -s -X POST "http://localhost:8000/api/v1/benchmark?iterations=20" | jq .
   ```
4. **Inspect Swagger Documentation**:
   Navigate to `http://localhost:8000/docs` in your browser.
