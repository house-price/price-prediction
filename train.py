"""
Offline training script: read dataset -> train -> save model file.

Usage:  python train.py
Run during Docker build so the image ships with a trained model (ready on start, reproducible).
"""
import pandas as pd

from app.config import settings
from app.model import PricePredictor


def main() -> None:
    df = pd.read_csv(settings.dataset_path)
    predictor = PricePredictor().train(df)
    predictor.save(settings.model_path)
    print(f"Model saved to {settings.model_path}")
    print(f"Training samples: {predictor.n_training_samples}")
    print(f"Metrics: {predictor.metrics}")
    print(f"Coefficients: {predictor.coefficients}")
    print(f"Intercept: {predictor.intercept}")


if __name__ == "__main__":
    main()
