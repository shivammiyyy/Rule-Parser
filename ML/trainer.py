"""
Training pipeline for all RAVEN ML models.

  train_all()  — trains OperatorClassifier, BoundaryDetector, BM25Autocomplete
               and saves each to disk under models/
"""

from pathlib import Path

from .models import BoundaryDetector, OperatorClassifier, BM25Autocomplete
from .data import (
    OPERATOR_TRAIN_DATA,
    BOUNDARY_SENTENCES,
    BOUNDARY_LABELS,
    BM25_CORPUS,
)
from .audit import AuditLogger


_MODELS_DIR = Path("models")
_audit = AuditLogger("logs/ml_audit.jsonl")


# ------------------------------------------------------------------ #
# Individual trainers                                                  #
# ------------------------------------------------------------------ #

def _train_operator_classifier() -> OperatorClassifier:
    print("[trainer] Training OperatorClassifier …")
    model = OperatorClassifier()
    model.fit(OPERATOR_TRAIN_DATA)
    model.save(str(_MODELS_DIR / "operator_classifier.pkl"))
    _audit.log(
        "operator_classifier",
        event="trained",
        samples=len(OPERATOR_TRAIN_DATA),
    )
    print(f"[trainer] OperatorClassifier saved → {_MODELS_DIR / 'operator_classifier.pkl'}")
    return model


def _train_boundary_detector() -> BoundaryDetector:
    print("[trainer] Training BoundaryDetector (CRF) …")
    model = BoundaryDetector()
    model.fit(BOUNDARY_SENTENCES, BOUNDARY_LABELS)
    model.save(str(_MODELS_DIR / "boundary_detector.pkl"))
    _audit.log(
        "boundary_detector",
        event="trained",
        samples=len(BOUNDARY_SENTENCES),
    )
    print(f"[trainer] BoundaryDetector saved → {_MODELS_DIR / 'boundary_detector.pkl'}")
    return model


def _train_autocomplete() -> BM25Autocomplete:
    print("[trainer] Building BM25Autocomplete index …")
    model = BM25Autocomplete()
    model.fit(BM25_CORPUS)
    model.save(str(_MODELS_DIR / "autocomplete.pkl"))
    _audit.log(
        "bm25_autocomplete",
        event="indexed",
        corpus_size=len(BM25_CORPUS),
    )
    print(f"[trainer] BM25Autocomplete saved → {_MODELS_DIR / 'autocomplete.pkl'}")
    return model


# ------------------------------------------------------------------ #
# Public entry point                                                   #
# ------------------------------------------------------------------ #

def train_all() -> dict:
    """
    Run the full training pipeline.

    Returns a dict with the three trained model instances:
        {
          "operator_classifier": OperatorClassifier,
          "boundary_detector":   BoundaryDetector,
          "autocomplete":        BM25Autocomplete,
        }
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    op_clf   = _train_operator_classifier()

    try:
        bd = _train_boundary_detector()
    except ImportError as e:
        print(f"[trainer] Skipping BoundaryDetector — {e}")
        bd = None

    try:
        ac = _train_autocomplete()
    except ImportError as e:
        print(f"[trainer] Skipping BM25Autocomplete — {e}")
        ac = None

    print("[trainer] ✓ train_all() complete.")
    return {
        "operator_classifier": op_clf,
        "boundary_detector":   bd,
        "autocomplete":        ac,
    }
