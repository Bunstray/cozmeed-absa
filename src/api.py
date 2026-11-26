"""Decoupled FastAPI service for Cozmeed ABSA single and batch scoring.

Exposes Pydantic schemas and an application factory (``create_app``) without
loading model weights at module import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

SentimentClass = Literal["Positive", "Negative", "None"]


class PredictRequest(BaseModel):
    """Request payload for single-review aspect sentiment prediction.

    Attributes:
        review_text: Indonesian product review text to score.
    """

    review_text: str = Field(..., min_length=1, description="Indonesian marketplace review text.")


class BatchPredictRequest(BaseModel):
    """Request payload for batch aspect sentiment prediction.

    Attributes:
        reviews: Non-empty list of Indonesian review strings to score.
    """

    reviews: list[str] = Field(..., min_length=1, description="List of review texts to classify.")


class PredictResponse(BaseModel):
    """Response payload containing predicted sentiments across all 4 aspects.

    Attributes:
        review_text: Original review text scored by the service.
        material_durability: Predicted sentiment for fabric, stitching, zippers, and durability.
        fit_comfort: Predicted sentiment for sizing, ergonomics, and comfort.
        price_value: Predicted sentiment for price and value for money.
        shipping_seller: Predicted sentiment for delivery, packaging, and seller service.
    """

    review_text: str
    material_durability: SentimentClass
    fit_comfort: SentimentClass
    price_value: SentimentClass
    shipping_seller: SentimentClass


class BatchPredictResponse(BaseModel):
    """Response payload wrapping ordered predictions for a batch request.

    Attributes:
        predictions: Ordered list of ``PredictResponse`` objects.
    """

    predictions: list[PredictResponse]


def create_app(model_dir: Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application for Cozmeed ABSA serving.

    Args:
        model_dir: Optional directory path for exported IndoBERT artifacts.
            When omitted, endpoint calls report that trained artifacts are not
            yet configured until Stage 5 serving setup.

    Returns:
        Configured ``FastAPI`` application instance with ``/health``,
        ``/predict``, and ``/predict/batch`` routes registered.

    Raises:
        None.

    Side Effects:
        Registers route handlers; deferred model loading will occur in lifespan
        startup once trained weights exist.

    Edge Cases:
        Importing this module or calling ``create_app()`` without trained weights
        succeeds cleanly so tests and OpenAPI inspection do not require GPU/model files.
    """
    app = FastAPI(
        title="Cozmeed ABSA API",
        version="0.1.0",
        description="CPU-efficient Aspect-Based Sentiment Analysis API for Cozmeed reviews.",
    )
    app.state.model_dir = model_dir

    @app.get("/health")
    def health_check() -> dict[str, str]:
        """Return basic service liveness and model configuration status."""
        configured = "configured" if app.state.model_dir is not None else "unconfigured"
        return {"status": "ok", "model_state": configured}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Score a single review across all four Cozmeed aspects."""
        raise HTTPException(
            status_code=501,
            detail="Inference endpoint awaits trained model weights in Stage 5.",
        )

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch_endpoint(request: BatchPredictRequest) -> BatchPredictResponse:
        """Score a batch of reviews across all four Cozmeed aspects."""
        raise HTTPException(
            status_code=501,
            detail="Batch inference endpoint awaits trained model weights in Stage 5.",
        )

    return app


app = create_app()
