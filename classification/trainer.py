"""
Standalone training script. Run separately from the main pipeline:
    python -m classification.trainer
"""

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold

from core.config import load_config, setup_logging
from preprocessing.preprocessor import NLPPreprocessor

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


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


def train_and_save(
    data_dir: Path = Path("data/train_data"),
    output_path: Path = Path("models/model.pkl"),
) -> None:
    """Full training workflow: load data, encode, cross-validate, train, save."""
    preprocessor = NLPPreprocessor()
    texts, labels = load_training_data(data_dir, preprocessor)

    if len(set(labels)) < 2:
        logger.error("Need at least 2 classes to train. Check training data files.")
        sys.exit(1)

    # Encode texts with sentence-transformer
    logger.info(f"Loading sentence-transformer model: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Encoding training texts...")
    embeddings = encoder.encode(texts, show_progress_bar=True)

    # Cross-validation on embeddings
    min_class_count = min(labels.count(label) for label in set(labels))
    n_splits = min(5, min_class_count)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")

    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, embeddings, labels, cv=cv, scoring="f1_weighted")
        logger.info(
            f"Cross-validation F1 (weighted): {scores.mean():.3f} (+/- {scores.std():.3f})"
        )
    else:
        logger.warning("Too few samples for cross-validation.")

    # Train on full dataset
    clf.fit(embeddings, labels)
    logger.info(f"Model trained. Classes: {list(clf.classes_)}")

    # Classification report on training data
    predictions = clf.predict(embeddings)
    report = classification_report(labels, predictions)
    logger.info(f"Training set classification report:\n{report}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "classifier": clf,
        "model_name": EMBEDDING_MODEL,
        "classes": list(clf.classes_),
    }
    joblib.dump(model_data, output_path)
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config)
    train_and_save(
        data_dir=Path("data/train_data"),
        output_path=config.model_path,
    )
