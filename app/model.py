"""
House price regression model wrapper.

Design notes:
  - This is a "pure logic" class with no FastAPI dependency, so it can be unit
    tested without the web layer.
  - Model = Pipeline of Ridge regression + StandardScaler:
      * the 7 features in this dataset correlate 0.98+ with each other (severe
        multicollinearity); plain OLS would flip coefficient signs and make
        them uninterpretable;
      * Ridge's L2 regularization stabilizes the coefficients; combined with
        standardization, coefficient magnitudes are directly comparable and can
        be shown as "feature importance" to model-info callers, with almost no
        loss of predictive power.
  - Honest metrics: first train/test split, compute r2/rmse/mae on the unseen
    test set; then refit on ALL data as the deployed model (small dataset, so
    we avoid wasting samples).
"""
from __future__ import annotations

import joblib
import os
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Fixed feature column order; assembled in this order at prediction time to avoid column misalignment
FEATURE_NAMES: list[str] = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]
TARGET = "price"
# Regularization strength; larger = smoother coefficients, more robust to collinearity
RIDGE_ALPHA = 10.0


def _build_pipeline() -> Pipeline:
    """
    Standardization + ridge regression. Pipeline fits/predicts the two steps as one model.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


class PricePredictor:
    def __init__(self) -> None:
        # leading underscore = conventional "private" field
        self._model: Pipeline | None = None
        self.feature_names: list[str] = list(FEATURE_NAMES)
        self.model_type = "Ridge (standardized features)"
        self.coefficients: dict[str, float] = {}
        self.intercept: float = 0.0
        self.metrics: dict[str, float] = {}
        self.n_training_samples: int = 0
        self.trained_at: str | None = None

    @property
    def is_ready(self) -> bool:
        """
        Whether the model is loaded/trained.
        """
        return self._model is not None

    def train(self, df: pd.DataFrame) -> "PricePredictor":
        X = df[self.feature_names]
        y = df[TARGET]

        # 1) Hold out 20% for honest evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # fixed seed for reproducibility
        )
        eval_model = _build_pipeline().fit(X_train, y_train)
        preds = eval_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5  # version-agnostic across sklearn
        self.metrics = {
            "r2": round(float(r2_score(y_test, preds)), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mean_absolute_error(y_test, preds)), 2),
        }

        # 2) Refit on ALL data as the final deployed model
        self._model = _build_pipeline().fit(X, y)
        ridge = self._model.named_steps["ridge"]  # extract the regression step from the pipeline
        # These are standardized coefficients: comparable, representing each feature's relative importance
        self.coefficients = {
            name: round(float(c), 4)
            for name, c in zip(self.feature_names, ridge.coef_)
        }
        self.intercept = round(float(ridge.intercept_), 4)  # ≈ mean training price
        self.n_training_samples = int(len(df))
        self.trained_at = datetime.now(timezone.utc).isoformat()
        return self

    def predict(self, rows: list[dict]) -> list[float]:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")
        # Build a DataFrame with named columns to keep column order and avoid sklearn feature-name warnings
        X = pd.DataFrame(rows, columns=self.feature_names)
        preds = self._model.predict(X)
        return [round(float(p), 2) for p in preds]

    # ---- Serialization: persist/restore training result (joblib suits numpy/sklearn objects) ----
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self.feature_names,
                "model_type": self.model_type,
                "coefficients": self.coefficients,
                "intercept": self.intercept,
                "metrics": self.metrics,
                "n_training_samples": self.n_training_samples,
                "trained_at": self.trained_at,
            },
            path,
        )

    def load(self, path: str) -> "PricePredictor":
        data = joblib.load(path)
        self._model = data["model"]
        self.feature_names = data["feature_names"]
        self.model_type = data["model_type"]
        self.coefficients = data["coefficients"]
        self.intercept = data["intercept"]
        self.metrics = data["metrics"]
        self.n_training_samples = data["n_training_samples"]
        self.trained_at = data["trained_at"]
        return self

    def load_or_train(self, model_path: str, dataset_path: str) -> "PricePredictor":
        """
        Called at startup: load the model file if present, otherwise train from the dataset and save.
        """
        if os.path.exists(model_path):
            return self.load(model_path)
        df = pd.read_csv(dataset_path)
        self.train(df)
        self.save(model_path)
        return self
