"""
Request / response data models (schemas).
"""
from pydantic import BaseModel, ConfigDict, Field


class HouseFeatures(BaseModel):
    """
    The 7 features of a single house. Field order/names must match the training columns.
    """

    # extra="forbid": reject unknown fields, so a misspelled field name is not silently ignored
    model_config = ConfigDict(extra="forbid")

    square_footage: float = Field(gt=0, le=100_000, examples=[1550], description="Living area (sq ft)")
    bedrooms: int = Field(ge=0, le=20, examples=[3], description="Number of bedrooms")
    bathrooms: float = Field(ge=0, le=20, examples=[2.0], description="Bathrooms (may be fractional, e.g. 2.5)")
    year_built: int = Field(ge=1800, le=2100, examples=[1997], description="Year built")
    lot_size: float = Field(gt=0, examples=[6800], description="Lot size")
    distance_to_city_center: float = Field(ge=0, examples=[4.1], description="Distance to city center")
    school_rating: float = Field(ge=0, le=10, examples=[7.6], description="School rating (0-10)")


class PredictRequest(BaseModel):
    """
    Prediction request: the instances array uniformly supports single and batch.
      - single: pass 1 element in instances
      - batch:  pass multiple elements in instances
    This is the industry-standard approach (e.g. Google AI Platform): one endpoint, both cases.
    """

    instances: list[HouseFeatures] = Field(min_length=1, description="One or more houses to predict")


class Prediction(BaseModel):
    predicted_price: float = Field(description="Predicted price")


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    count: int = Field(description="Number of predictions in this response")


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    features: list[str]
    coefficients: dict[str, float] = Field(description="Standardized coefficients (comparable feature importance)")
    intercept: float
    metrics: dict[str, float] = Field(description="Held-out test-set metrics: r2 / rmse / mae")
    n_training_samples: int
    trained_at: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
