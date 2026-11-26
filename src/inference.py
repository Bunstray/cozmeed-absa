"""CPU-only inference runtime for Cozmeed Aspect-Based Sentiment Analysis.

Loads fine-tuned IndoBERT weights once on ``device="cpu"`` and executes
predictions inside ``torch.inference_mode()`` for both FastAPI and Streamlit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AspectPrediction:
    """Structured 4-aspect sentiment prediction for a single review.

    Attributes:
        review_text: Input review string scored by the model.
        aspects: Mapping from each aspect in ``src.ASPECTS`` to a predicted
            label in ``src.SENTIMENT_LABELS``.
    """

    review_text: str
    aspects: dict[str, str]


class AspectSentimentPredictor:
    """CPU-only wrapper around a serialized multi-aspect IndoBERT classifier.

    Attributes:
        model_dir: Directory containing exported tokenizer and model weights.
        device: Target PyTorch execution device, fixed to ``"cpu"`` locally.
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        """Initialize the predictor configuration without eager heavy I/O.

        Args:
            model_dir: Directory containing serialized model weights under
                ``models/indobert_absa/``.
            device: Execution device string. Must remain ``"cpu"`` in local
                environments.

        Raises:
            ValueError: If ``device`` is empty or unsupported locally.

        Side Effects:
            Stores configuration attributes; model loading occurs in ``load()``.

        Edge Cases:
            Defers disk loading until ``load()`` so importing ``src.inference``
            succeeds even before trained weights are present.
        """
        if not device:
            raise ValueError("device must be a non-empty string.")
        self.model_dir = Path(model_dir)
        self.device = device
        self._is_loaded = False

    def load(self) -> None:
        """Load tokenizer and model weights onto CPU in evaluation mode.

        Raises:
            FileNotFoundError: If ``self.model_dir`` does not contain trained
                model artifacts.

        Side Effects:
            Loads model weights into memory once for reuse across requests.

        Edge Cases:
            Repeated calls should be idempotent once weights are loaded.
        """
        raise NotImplementedError("Model loading awaits exported weights in models/indobert_absa/.")

    def predict(self, review_text: str) -> AspectPrediction:
        """Score a single Indonesian review across all four Cozmeed aspects.

        Args:
            review_text: Raw or normalized Indonesian review text.

        Returns:
            ``AspectPrediction`` mapping each aspect in ``src.ASPECTS`` to
            ``Positive``, ``Negative``, or ``None``.

        Raises:
            RuntimeError: If called before model weights are loaded.

        Side Effects:
            Runs PyTorch forward pass inside ``torch.inference_mode()``.

        Edge Cases:
            Empty or whitespace-only inputs should be validated cleanly.
        """
        raise NotImplementedError("Inference awaits exported weights in models/indobert_absa/.")

    def predict_batch(self, review_texts: Sequence[str], batch_size: int = 16) -> list[AspectPrediction]:
        """Score a sequence of Indonesian reviews on CPU in bounded batches.

        Args:
            review_texts: Sequence of review strings to classify.
            batch_size: Maximum number of texts per forward pass.

        Returns:
            List of ``AspectPrediction`` objects preserving input order.

        Raises:
            ValueError: If ``batch_size < 1``.
            RuntimeError: If called before model weights are loaded.

        Side Effects:
            Executes batched forward passes inside ``torch.inference_mode()``.

        Edge Cases:
            Empty sequences return an empty list without invoking the model.
        """
        raise NotImplementedError("Batch inference awaits exported weights in models/indobert_absa/.")


def load_predictor(model_dir: Path, device: str = "cpu") -> AspectSentimentPredictor:
    """Instantiate and load a singleton CPU ``AspectSentimentPredictor``.

    Args:
        model_dir: Directory containing exported IndoBERT artifacts.
        device: Target PyTorch device (defaults to ``"cpu"``).

    Returns:
        Loaded ``AspectSentimentPredictor`` ready for inference.

    Raises:
        FileNotFoundError: If ``model_dir`` artifacts are missing.

    Side Effects:
        Loads tokenizer and model weights onto CPU memory.

    Edge Cases:
        Propagates missing-weight errors cleanly for API/UI startup checks.
    """
    predictor = AspectSentimentPredictor(model_dir=model_dir, device=device)
    predictor.load()
    return predictor


def predict_single(predictor: AspectSentimentPredictor, review_text: str) -> AspectPrediction:
    """Convenience helper to score a single review with an active predictor.

    Args:
        predictor: Loaded ``AspectSentimentPredictor`` instance.
        review_text: Input review text.

    Returns:
        ``AspectPrediction`` for the provided review.

    Raises:
        RuntimeError: If ``predictor`` is not loaded.

    Side Effects:
        Delegates to ``predictor.predict``.

    Edge Cases:
        See ``AspectSentimentPredictor.predict``.
    """
    return predictor.predict(review_text)


def predict_batch(
    predictor: AspectSentimentPredictor,
    review_texts: Sequence[str],
    batch_size: int = 16,
) -> list[AspectPrediction]:
    """Convenience helper to score multiple reviews with an active predictor.

    Args:
        predictor: Loaded ``AspectSentimentPredictor`` instance.
        review_texts: Sequence of input review texts.
        batch_size: Batch size for CPU inference.

    Returns:
        Ordered list of ``AspectPrediction`` records.

    Raises:
        ValueError: If ``batch_size < 1``.
        RuntimeError: If ``predictor`` is not loaded.

    Side Effects:
        Delegates to ``predictor.predict_batch``.

    Edge Cases:
        Empty sequences return ``[]``.
    """
    return predictor.predict_batch(review_texts, batch_size=batch_size)
