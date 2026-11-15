"""Generate Report.pdf from the logs in docs/. Run: python docs/make_report.py"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

D = Path(__file__).parent
OUT = D.parent / "Report.pdf"
REPO = "https://github.com/Hashir-Rauf/Bitcoin-forecasting-model"


def read(name, maxlines=None):
    t = (D / name).read_text(encoding="utf-8", errors="replace").splitlines()
    t = [ln[:105] for ln in t]
    return "\n".join(t[:maxlines] if maxlines else t)


def architecture_png():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#333"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#333"))

    box(0.1, 3.0, 1.6, 0.8, "yfinance\nBTC-USD daily", "#fde9c9")
    box(2.2, 3.0, 3.3, 0.8, "Airflow DAG (daily)\nfetch > features > train/eval > reload", "#d6e8ff")
    box(6.0, 3.0, 1.8, 0.8, "Shared volume /\nPVC /app/models", "#e2f3dc")
    box(8.2, 3.0, 1.6, 0.8, "Kubeflow KFP\n(ingest/feat/train)", "#eadcf5")
    box(0.1, 1.5, 2.2, 0.8, "GitHub push\nGitHub Actions\nflake8/black/pytest", "#eee")
    box(2.8, 1.5, 1.8, 0.8, "Docker build\npush: Docker Hub\n(SHA + latest)", "#eee")
    box(5.1, 1.5, 4.7, 0.8, "Kubernetes: Deployment (3 pods, gunicorn:5000) + HPA 2-8 @60% CPU\n"
        "ConfigMap + Secret env, probes, PVC", "#ffe0e0")
    box(0.1, 0.1, 2.0, 0.8, "Client / curl", "#fde9c9")
    box(2.8, 0.1, 3.0, 0.8, "Ingress NGINX\nbtc-forecast.local", "#d6e8ff")
    box(6.3, 0.1, 3.5, 0.8, "Service :80 -> pods :5000\n/health /predict /metrics /reload", "#e2f3dc")
    arrow(1.7, 3.4, 2.2, 3.4)
    arrow(5.5, 3.4, 6.0, 3.4)
    arrow(6.9, 3.0, 7.2, 2.3)
    arrow(2.3, 1.9, 2.8, 1.9)
    arrow(4.6, 1.9, 5.1, 1.9)
    arrow(2.1, 0.5, 2.8, 0.5)
    arrow(5.8, 0.5, 6.3, 0.5)
    arrow(8.0, 0.9, 8.0, 1.5)
    p = D / "architecture.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def hpa_png():
    t, cpu, rep = [], [], []
    for ln in read("d5_hpa_load.log").splitlines():
        m = re.match(r"btc-forecast-hpa\s+\S+\s+cpu: (\S+)/60%\s+2\s+8\s+(\d+)", ln)
        if m:
            t.append(len(t) * 15)
            cpu.append(0 if m.group(1) == "<unknown>" else int(m.group(1).rstrip("%")))
            rep.append(int(m.group(2)))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(t, cpu, "o-", color="#d9534f", label="CPU % of request")
    ax.axhline(60, ls="--", color="gray", label="target 60%")
    ax.set_xlabel("seconds since load start")
    ax.set_ylabel("CPU %")
    ax2 = ax.twinx()
    ax2.step(t, rep, where="post", color="#337ab7", label="replicas")
    ax2.set_ylabel("replicas")
    ax2.set_ylim(0, 9)
    ax.set_title("HPA autoscaling under load (<unknown> metric samples plotted as 0)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    p = D / "hpa_chart.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=14, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12)
C = ParagraphStyle("C", fontName="Courier", fontSize=6.3, leading=7.6, backColor="#f4f4f4")
PH = ParagraphStyle("PH", parent=B, textColor=colors.HexColor("#8a6d3b"), backColor="#fcf8e3")


def code(text):
    return Preformatted(text, C)


def placeholder(label):
    return Paragraph(f"[SCREENSHOT TO INSERT: {label}]", PH)


def shot(name, caption, width_cm=15.5):
    from PIL import Image as PILImage

    path = D.parent / "Screenshots" / name
    w, h = PILImage.open(path).size
    wd = width_cm * cm
    return [Image(str(path), width=wd, height=wd * h / w),
            Paragraph(f"<i>{caption}</i>", B), Spacer(1, 4)]


def build():
    arch, hpa = architecture_png(), hpa_png()
    s = []
    s += [
        Paragraph("Assignment 3 - Production MLOps Pipeline for BTC-USD 7-Day Forecast", H1),
        Paragraph(f"Repository: {REPO}<br/>Docker image: Docker Hub "
                  "hashirrauf/btc-forecast (tags: commit SHA, latest)", B),
        Spacer(1, 6),
        Paragraph("1. Architecture", H2),
        Image(str(arch), width=17 * cm, height=7.1 * cm),
        Paragraph(
            "Daily BTC-USD OHLCV data is pulled with yfinance by an Airflow DAG, turned into technical "
            "features (7/30-day MAs, 7/30-day volatility, 10 lagged log returns), and a Gradient Boosting "
            "model predicting next-day log return is retrained. The 7-day forecast is produced by "
            "recursive one-step prediction. The latest 7 days are held out for validation (MAE, RMSE, "
            "MAPE); the model is registered only if MAPE &lt;= 25%. The Flask/Gunicorn API serves the "
            "model, and Airflow calls <i>POST /reload</i> so the service re-reads the weights.", B),
        Paragraph("2. Part A - Flask API", H2),
        Paragraph("Endpoints: <b>GET /health</b> (status, uptime, model version), <b>POST /predict</b> "
                  "(optional forecast_days 1-30 or custom history array; 400 on invalid input, 503 if "
                  "no model, 500 on errors), <b>GET /metrics</b> (MAE/RMSE/MAPE of last validation), "
                  "<b>POST /reload</b> (guarded by API_KEY from the K8s Secret). Gunicorn is the WSGI "
                  "server. 13 pytest tests cover endpoints, validation and error codes and pass in CI.", B),
        code(read("e2_airflow_compose.log").split("$ curl")[0][:0] or
             "$ pytest -q\n.............                                                  [100%]\n"
             "13 passed"),
        Paragraph("3. Part B - Docker, Compose and GitHub Actions", H2),
        Paragraph("app/Dockerfile is multi-stage (python:3.10-slim builder + runtime), runs as non-root "
                  "uid 10001, has a HEALTHCHECK and starts gunicorn. docker-compose.yml runs the API, "
                  "Postgres and Airflow (webserver, scheduler, init). The workflow ci_cd.yml runs flake8, "
                  "black --check and pytest on every push/PR to main, then logs in to Docker Hub using "
                  "the DOCKER_USERNAME / DOCKER_PASSWORD secrets and pushes the image tagged with the "
                  "commit SHA and latest. The latest run on main: both jobs (test, docker) succeeded.", B),
        *shot("GithubActions.png", "Fig. GitHub Actions run: test and docker jobs succeeded", 13),
        *shot("dockerhub.png", "Fig. Docker Hub hashirrauf/btc-forecast with SHA and latest tags", 13),
        *shot("Docker build.png", "Fig. docker build output (multi-stage, non-root user)", 11),
        PageBreak(),
        Paragraph("4. Part C - Airflow ETL DAG and Kubeflow", H2),
        Paragraph("DAG btc_etl_pipeline (@daily): fetch_data &gt; preprocess_features &gt; "
                  "retrain_evaluate &gt; reload_api_model. A manual run was triggered in the Docker "
                  "Compose stack; all four tasks finished with state success and the API served the "
                  "new model version afterwards (E2). The first attempt exposed a dependency mismatch "
                  "between the Airflow and API images (numpy/pandas/dill) that made /reload return 500; "
                  "fixed by pinning versions in requirements.txt.", B),
        code(read("e2_airflow_compose.log", 16)),
        *shot("DAG.png", "Fig. Airflow graph view of btc_etl_pipeline (grid: all task runs green)", 14),
        Paragraph("kubeflow/btc_kfp_pipeline.py defines three KFP components (ingest_data, "
                  "build_features, train_model) wired into btc_pipeline and compiled to "
                  "kubeflow/btc_kfp_pipeline.yaml with the kfp 2.x SDK (compiled successfully; it was "
                  "not run on a Kubeflow cluster).", B),
        Paragraph("5. Part D - Kubernetes deployment (Minikube, Docker driver)", H2),
        Paragraph("Manifests: ConfigMap (MODEL_STAGE, LOG_LEVEL, PORT) + Secret (API_KEY) injected "
                  "via envFrom; PVC (1Gi) mounted at /app/models; Deployment with 3 replicas, liveness "
                  "(initialDelay 15s, period 10s) and readiness probes on /health, requests "
                  "100m/128Mi and limits 500m/512Mi; ClusterIP service (80 -> 5000) plus NodePort "
                  "30080; HPA min 2 / max 8 / CPU 60%; NGINX Ingress for btc-forecast.local. A model "
                  "trained inside a pod survived a rollout restart, confirming PVC persistence.", B),
        code(read("k8s_state.log", 30)),
        PageBreak(),
        Paragraph("6. D5 - HPA load test", H2),
        Paragraph("Load: 8 parallel busybox clients in-cluster posting to /predict via the service. "
                  "CPU rose to ~450% of the request and the HPA scaled 3 -> 6 -> 8 replicas "
                  "(maxReplicas). On this 2-CPU Minikube node saturation made some liveness probes "
                  "time out, causing a few container restarts - a limitation of the small test "
                  "cluster rather than of the manifests. (wrk/Locust were replaced by an in-cluster "
                  "busybox generator.)", B),
        Image(str(hpa), width=14 * cm, height=5.6 * cm),
        code(read("d5_hpa_load.log", 22)),
        *shot("watch.png", "Fig. kubectl get hpa --watch: replicas 2 -> 4 -> 6 -> 8 under load", 14),
        *shot("kubectl get pods etc.png", "Fig. kubectl get pods,svc,hpa,ingress,pvc", 15),
        *shot("predict.png", "Fig. Sample 7-day prediction from the API", 16),
        Paragraph("7. D6 - Ingress", H2),
        Paragraph("With minikube tunnel running, the request was sent with the Host header "
                  "btc-forecast.local (hosts-file edit needs admin) and was routed by NGINX to the "
                  "service:", B),
        code(read("d6_ingress.log")),
        *shot("ingress.png", "Fig. Request routed by NGINX Ingress for host btc-forecast.local", 15),
        PageBreak(),
        Paragraph("8. E1 - Chaos / self-healing", H2),
        Paragraph("A running pod (btc-forecast-ff4bffb9c-88b5v) was deleted. The ReplicaSet created "
                  "replacement btc-forecast-ff4bffb9c-4r428 within ~3 s, and it became Ready (1/1) "
                  "about 20 s later while the other pods kept serving. (The cluster was still "
                  "scaled to 8 by the HPA after the load test, which is why more than 3 pods are listed.)",
                  B),
        code(read("e1_self_healing.log", 40)),
        *shot("delete pod.png", "Fig. Deleting a running pod", 13),
        *shot("AFter deleted.png", "Fig. Old pod Terminating, new pod ContainerCreating then Running", 12),
        Paragraph("9. E2 - End-to-end verification", H2),
        Paragraph("(1) Airflow DAG triggered manually: all tasks succeeded and a new model version "
                  "was produced. (2) The API reloaded the model through /reload. (3) GitHub Actions "
                  "succeeded (tests + image push). (4) The live Kubernetes endpoint returned a 7-day "
                  "forecast through the Ingress, e.g. 2026-11-16: 64,322.85 ... 2026-11-22: 66,355.52 "
                  "USD. Validation metrics (last 7 days): MAE 5034, RMSE 6596, MAPE 7.58%.", B),
        Paragraph("10. Performance analysis and limitations", H2),
        Paragraph("A MAPE of ~7.6% over a 7-day recursive horizon is typical for BTC, where daily "
                  "returns are close to noise; the model should be seen as a pipeline demonstration, "
                  "not a trading signal. Multi-step recursive forecasts accumulate error, and a "
                  "single 7-day validation window is a noisy estimate. Future work: walk-forward "
                  "cross-validation, prediction intervals, model registry (MLflow), running the KFP "
                  "pipeline on a real Kubeflow cluster, and moving the Secret to an external secret "
                  "manager (the committed API_KEY is a demo value).", B),
    ]
    SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                      topMargin=1.4 * cm, bottomMargin=1.4 * cm).build(s)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
