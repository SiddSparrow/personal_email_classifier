import logging
from pathlib import Path

import joblib

from core.interfaces import EmailClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class LogisticRegressionClassifier(EmailClassifier):
    """Classifies text using a pre-trained scikit-learn Pipeline (TF-IDF + LogisticRegression)."""

    def __init__(self, model_path: Path, confidence_threshold: float = 0.75) -> None:
        self._threshold = confidence_threshold
        logger.info(f"Loading model from {model_path}")
        self._pipeline = joblib.load(model_path)
        self._classes = list(self._pipeline.classes_)
        logger.info(f"Model loaded. Classes: {self._classes}")

    def classify(self, text: str) -> ClassificationResult:
        probas = self._pipeline.predict_proba([text])[0]
        predicted_idx = probas.argmax()
        category = self._classes[predicted_idx]
        confidence = float(probas[predicted_idx])

        is_positive = category == "job_offer" and confidence >= self._threshold

        logger.debug(
            f"Classification: {category} ({confidence:.1%}), "
            f"positive={is_positive}, threshold={self._threshold:.1%}"
        )
        return ClassificationResult(
            category=category,
            confidence=confidence,
            is_positive=is_positive,
        )
