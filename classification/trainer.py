"""
Standalone training script. Run separately from the main pipeline:
    python -m classification.trainer
"""

import logging
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from core.config import load_config, setup_logging
from preprocessing.preprocessor import NLPPreprocessor

logger = logging.getLogger(__name__)


def load_training_data(
    data_dir: Path,
    preprocessor: NLPPreprocessor,
) -> tuple[list[str], list[str]]:
    """
    Load and preprocess training data from text files.
    Each file = one label. Filename (minus trailing 's') becomes the label.
    One training example per line.
    """
    texts: list[str] = []
    labels: list[str] = []

    for txt_file in sorted(data_dir.glob("*.txt")):
        label = txt_file.stem.rstrip("s")  # "job_offers" -> "job_offer"
        lines = txt_file.read_text(encoding="utf-8").strip().splitlines()
        logger.info(f"Loaded {len(lines)} samples for '{label}' from {txt_file.name}")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            cleaned = preprocessor.preprocess(subject="", body_html="", body_plain=line)
            if cleaned:
                texts.append(cleaned)
                labels.append(label)

    logger.info(f"Total training samples: {len(texts)}")
    return texts, labels


def build_sklearn_pipeline() -> Pipeline:
    """Build TF-IDF + Naive Bayes pipeline as specified in doc."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def train_and_save(
    data_dir: Path = Path("data/train_data"),
    output_path: Path = Path("models/model.pkl"),
) -> None:
    """Full training workflow: load data, cross-validate, train, save."""
    preprocessor = NLPPreprocessor(use_stemmer=True)
    texts, labels = load_training_data(data_dir, preprocessor)

    if len(set(labels)) < 2:
        logger.error("Need at least 2 classes to train. Check training data files.")
        sys.exit(1)

    pipeline = build_sklearn_pipeline()

    # Cross-validation
    min_class_count = min(labels.count(label) for label in set(labels))
    n_splits = min(5, min_class_count)

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, texts, labels, cv=cv, scoring="f1_weighted")
        logger.info(
            f"Cross-validation F1 (weighted): {scores.mean():.3f} (+/- {scores.std():.3f})"
        )
    else:
        logger.warning("Too few samples for cross-validation.")

    # Train on full dataset
    pipeline.fit(texts, labels)
    logger.info(f"Model trained. Classes: {list(pipeline.classes_)}")

    # Classification report on training data (for quick sanity check)
    predictions = pipeline.predict(texts)
    report = classification_report(labels, predictions)
    logger.info(f"Training set classification report:\n{report}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    train_and_save(
        data_dir=Path("data/train_data"),
        output_path=config.model_path,
    )
