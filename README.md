# BTC-USD 7-Day Forecast — Production MLOps Pipeline

Flask + Gunicorn API, Docker, GitHub Actions CI/CD, Airflow ETL, Kubeflow Pipelines, Kubernetes.

```
yfinance -> Airflow DAG (fetch -> features -> retrain/eval -> /reload) -> shared volume / PVC
                                                                         |
GitHub Actions (flake8, black, pytest, docker build+push) -> registry -> K8s Deployment (3 pods)
Client -> Ingress (btc-forecast.local) -> Service :80 -> pods :5000 (gunicorn) ; HPA 2..8 @60% CPU
```

## Layout
| Path | Role |
|---|---|
| `app/app.py`, `app/Dockerfile` | API (`/health`, `/predict`, `/metrics`, `/reload`) + multi-stage non-root image |
| `src/` | `etl_fetch.py`, `train_forecast.py`, `forecast_lib.py` (features + recursive forecast) |
| `dags/btc_etl_pipeline.py` | Daily Airflow DAG |
| `kubeflow/btc_kfp_pipeline.py` | KFP components + pipeline (compiles to YAML) |
| `k8s/` | configmap-secret, pvc, deployment, service, hpa, ingress |
| `.github/workflows/ci_cd.yml` | Lint + test, then build/push to Docker Hub (needs `DOCKER_USERNAME`, `DOCKER_PASSWORD` secrets) |

Model: Gradient Boosting on lagged log-returns, MA ratios and volatility; recursive 7-day forecast.
Validation = last 7 days held out (MAE/RMSE/MAPE); the model is registered only if MAPE <= 25%.

## Run locally
```bash
pip install -r requirements-dev.txt
python -m src.etl_fetch && python -m src.train_forecast
pytest -q
MODEL_DIR=models gunicorn --bind 0.0.0.0:5000 app.app:app
curl -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{"forecast_days":7}'
```

## Docker Compose (API + Airflow + Postgres)
```bash
docker compose up --build        # Airflow UI :8080 (admin/admin), API :5000
```

## Kubernetes (Minikube)
```bash
minikube start && minikube addons enable ingress && minikube addons enable metrics-server
docker build -f app/Dockerfile -t btc-forecast:latest . && minikube image load btc-forecast:latest
kubectl apply -f k8s/configmap-secret.yaml -f k8s/pvc.yaml -f k8s/deployment.yaml \
              -f k8s/service.yaml -f k8s/hpa.yaml -f k8s/ingress.yaml
# train a model into the PVC (one-off):
kubectl exec deploy/btc-forecast -- python -c "from src.etl_fetch import main; main('/tmp/r.csv')"
kubectl exec deploy/btc-forecast -- python -m src.train_forecast --raw /tmp/r.csv --model-dir /app/models
kubectl rollout restart deploy/btc-forecast
echo "$(minikube ip) btc-forecast.local" >> /etc/hosts
curl http://btc-forecast.local/predict -X POST
# E1 self-healing
kubectl delete pod <pod>; kubectl get pods -w
# D5 load test
kubectl get hpa --watch &  wrk -t4 -c100 -d120s -s post.lua http://btc-forecast.local/predict
```
Compile the KFP spec: `pip install kfp && python kubeflow/btc_kfp_pipeline.py`.
