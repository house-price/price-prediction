# price-prediction

House price prediction microservice built with FastAPI + scikit-learn (Ridge regression).

## Endpoints

| Method | Path          | Description                                       |
|--------|---------------|---------------------------------------------------|
| GET    | `/health`     | Health check (for K8s probes)                     |
| POST   | `/predict`    | Single/batch price prediction (`instances` array) |
| GET    | `/model-info` | Model coefficients and performance metrics        |
| GET    | `/docs`       | Swagger UI (auto-generated, for live demos)       |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py                  # train and produce models/model.joblib
uvicorn app.main:app --reload    # serve at http://localhost:8000/docs
```

## Test

```bash
python -m pytest   # 13 tests: unit (model logic) + integration (HTTP API)
```

## Docker

```bash
docker build -t price-prediction:1.0.0 .
docker run -p 8000:8000 price-prediction:1.0.0
```

## Example call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances":[
        {"square_footage":1550,"bedrooms":3,"bathrooms":2,"year_built":1997,
         "lot_size":6800,"distance_to_city_center":4.1,"school_rating":7.6}
      ]}'
```

## Design notes

- **Model choice**: the 7 features correlate 0.98+ with each other (multicollinearity), so
  plain OLS produces unreliable coefficient signs. We use **Ridge + StandardScaler**: predictive
  power is essentially unchanged (R²≈0.97) and the standardized coefficients serve as comparable
  feature importances.
- **Metrics**: `r2 / rmse / mae` computed on a held-out test split; final model refit on all data.
- **Layering**: `model.py` is pure business logic (independently unit-testable); `main.py` is a thin web layer.
- **Reproducibility**: fixed random seed; pinned dependencies; model trained at image build time.
