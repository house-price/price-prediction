"""
FastAPI application entry point (≈ Spring Boot main class + @RestController).

After startup, open http://localhost:8000/docs for the auto-generated Swagger UI.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import PricePredictor
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

# Singleton: one model instance shared by the whole process (linear prediction is stateless & thread-safe)
predictor = PricePredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # -- Startup phase: load or train the model --
    predictor.load_or_train(settings.model_path, settings.dataset_path)
    yield
    # -- Shutdown phase: no resources to release here --


app = FastAPI(
    title="House Price Prediction API",
    description="House price prediction service based on a scikit-learn Ridge regression",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow CORS so the frontend/BFF can call directly and for live demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """
    Health check: used for liveness; K8s liveness/readiness probes hit this.
    """
    return HealthResponse(status="ok", model_loaded=predictor.is_ready)


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """
    Price prediction: 1 instance = single prediction, multiple = batch.
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # model_dump() turns a Pydantic object into a plain dict
    rows = [f.model_dump() for f in req.instances]
    prices = predictor.predict(rows)
    return PredictResponse(
        predictions=[Prediction(predicted_price=p) for p in prices],
        count=len(prices),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["inference"])
def model_info() -> ModelInfoResponse:
    """
    Return model coefficients and performance metrics for callers/monitoring.
    """
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=predictor.model_type,
        features=predictor.feature_names,
        coefficients=predictor.coefficients,
        intercept=predictor.intercept,
        metrics=predictor.metrics,
        n_training_samples=predictor.n_training_samples,
        trained_at=predictor.trained_at or "",
    )
