# Aperture — Climate Visibility Station

An end-to-end system that predicts maximum visibility distance from live
weather readings, built on top of your `cleaned_data.csv` and the project
brief in the slides you provided.

```
climate-visibility/
├── training/
│   ├── train_model.py       ← run this in VS Code to train the model
│   └── cleaned_data.csv
├── model/                     ← created by train_model.py
│   ├── visibility_model.joblib
│   ├── metrics.json
│   └── eda_*.png / feature_importance.png
├── backend/
│   ├── app.py                ← FastAPI service, serves /api/predict
│   └── requirements.txt
└── frontend/
    └── index.html             ← single-file, no build step, fully responsive
```

## 1. Train the model (VS Code, Python)

```bash
cd training
pip install pandas numpy seaborn matplotlib scikit-learn xgboost joblib
python train_model.py
```

What it does:
- Loads `cleaned_data.csv` and runs EDA (correlation heatmap, distributions,
  visibility-vs-humidity scatter — saved as PNGs in `model/`).
- Engineers domain features: dew-point spread, a fog-likelihood proxy,
  approximate air density, and cyclical month/hour encodings.
- Trains **Linear Regression**, **Random Forest**, and **XGBoost**, prints
  RMSE / MAE / R² for each, and keeps the best one.
- On your dataset, results were:

  | Model | RMSE | MAE | R² |
  |---|---|---|---|
  | Linear Regression | 1.65 | 0.98 | 0.45 |
  | Random Forest | 1.23 | 0.51 | 0.70 |
  | **XGBoost (selected)** | **1.22** | **0.51** | **0.70** |

- Exports one bundle, `model/visibility_model.joblib`, containing the model,
  scaler, feature list and metrics — the backend just loads this one file.

Re-run any time you get more data; the backend automatically picks up the
new `.joblib` bundle on restart.

## 2. Run the backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `POST /api/predict` — takes the raw weather readings and returns
  visibility (km + miles), a classified "scene" (sunny / cloudy / rain /
  fog / snow / storm / clear-night / cloudy-night), a severity level, and
  which model produced the number.
- `GET /api/health` — simple check the frontend uses to detect the API.
- `GET /docs` — interactive Swagger UI for testing requests by hand.

The scene classification uses the same inputs as the model (humidity, dew
point spread, precipitation, wind, hour-of-day) so the frontend's visuals
always match the physical conditions you entered, not just the predicted
number.

## 3. Run the frontend

It's a single static file — no npm install, no build step.

```bash
cd frontend
python3 -m http.server 8080
# open http://localhost:8080
```

By default it talks to `http://localhost:8000`. To point it at a deployed
backend, either:
- edit the `API_BASE` constant near the top of the `<script>` block, or
- inject it before load: `<script>window.API_BASE = "https://your-api.com";</script>`

If the backend is unreachable, the page automatically falls back to a
rough client-side estimate so the demo still works — a status line at the
bottom tells you which mode is active.

### What makes it "alive"
- A circular instrument gauge (like an aircraft visibility dial) sweeps to
  the predicted distance with an eased spring animation.
- The sky viewport behind it is a canvas particle system that swaps
  entirely per condition: drifting clouds, streaking rain, drifting
  snowflakes, rolling fog bands, and randomized lightning flashes for
  storms — plus a starfield/moon for night scenes.
- Every input is a slider with live value readout, and the whole scene
  re-themes in real time as you drag — you see the "weather" change before
  you even press the button.
- Fully responsive: single-column console on phones, fluid gauge sizing,
  `prefers-reduced-motion` respected, visible keyboard focus states.

## 4. Deploying (any platform)

**Backend** — any container/Python host works (Render, Railway, Fly.io,
an EC2/Azure box per your own architecture slides, etc.):
```bash
# minimal Dockerfile you can drop into backend/
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ../model /app/model
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Frontend** — since it's one static HTML file, it deploys anywhere static
files are served: Netlify, Vercel, GitHub Pages, S3+CloudFront, or the same
box as the backend behind Nginx. Just set `API_BASE` to your backend's
public URL before deploying.

Because both pieces are plain HTTP, the same setup works identically on
desktop, mobile browsers, or embedded in a webview — nothing here depends
on a particular OS or device.
