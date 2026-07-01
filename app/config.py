"""
Application configuration.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() disables pydantic's protected-warning for "model_" fields
    model_config = SettingsConfigDict(env_prefix="ML_", protected_namespaces=())

    # Path to the trained model file (joblib-serialized)
    model_path: str = "models/model.joblib"
    # Path to the training dataset (used to train on first start if no model file exists)
    dataset_path: str = "data/House_Price_Dataset.csv"
    # Whether to allow CORS (needed when the frontend/BFF calls directly from the browser)
    cors_allow_origins: list[str] = ["*"]


settings = Settings()
