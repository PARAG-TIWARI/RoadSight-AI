# Local development and deployment

This guide covers local development and container use for the current repository. The deployment examples are starting points; they are not evidence that a hosted service is currently running.

## Prerequisites

- Python 3.11 (recommended; the backend Docker image uses Python 3.11).
- Node.js 20 LTS and npm.
- Git.
- Docker Engine and Docker Compose plugin, if using containers.

The ONNX Runtime path is the default backend setup. The optional PyTorch path additionally needs `torch` and `ultralytics`.

## 1. Run the backend locally

From the repository root:

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy the root environment template into the backend working directory:

**Windows PowerShell**

```powershell
Copy-Item ../.env.example .env
```

**macOS / Linux**

```bash
cp ../.env.example .env
```

The default template selects ONNX inference and the `best.onnx` artifact. Start FastAPI from the `backend/` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Useful local endpoints:

- Health: `http://localhost:8000/api/v1/health`
- Model information: `http://localhost:8000/api/v1/model`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 2. Run the frontend locally

Open a second terminal:

```bash
cd frontend
npm ci
```

Create `frontend/.env.local`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Start the Vite development server:

```bash
npm run dev
```

Open `http://localhost:5173`.

## 3. Run with Docker Compose

From the repository root:

```bash
docker compose up --build
```

The Compose configuration publishes:
- Frontend: `http://localhost:80`
- Backend: `http://localhost:8000`

The backend container is configured to listen on port 8000 in Compose so that its published port and health check use the same container port.

Stop the stack with:

```bash
docker compose down
```

## 4. Run tests and build checks

Backend tests (from the repository root, with backend dependencies installed):

```bash
python -m pytest backend/tests -v
```

Frontend checks (from `frontend/`):

```bash
npm run lint
npm run build
```

Record the actual command output and environment when reporting test results.

## 5. Hosted deployment considerations

Before deploying a public instance, review the following:

- Use HTTPS for the frontend-to-backend connection.
- Set `CORS_ORIGINS` to the exact frontend origin(s); do not use a wildcard for a credentialed or sensitive deployment.
- Set the backend `PORT` to the port supplied by the hosting platform.
- Keep `MODEL_FORMAT=onnx` unless the environment has been provisioned with the optional PyTorch dependencies and sufficient memory.
- Start with one server worker. Each worker loads its own model instance, so increasing worker count increases model memory use; scale only after measuring the available RAM and throughput.
- Use persistent storage only if processed videos must survive process restarts. Current video outputs are stored in the configured temporary directory and may be removed.
- Apply an ingress/request-body limit at the proxy or hosting layer. The application currently reads the upload payload before its file-size validation runs.
- Store secrets in the hosting provider's environment configuration. Do not commit production `.env` files or model/API credentials.

The backend Docker image uses a default port of 10000 when no `PORT` is supplied. Docker Compose overrides that to 8000 to match its local port mapping. For other platforms, use the platform-provided port.

## 6. Operational checks

After deployment, verify the service using the platform's assigned URL:

```bash
curl -i https://<backend-host>/api/v1/health
curl -i https://<backend-host>/api/v1/model
```

Run a small benchmark only when the service is not handling latency-sensitive traffic:

```bash
curl -i "https://<backend-host>/api/v1/benchmark?iterations=10"
```

The benchmark endpoint runs synchronously in the API process and can consume CPU/GPU resources while it is running.

A health response confirms that the API process is responding; it does not establish model accuracy or end-to-end application correctness.
