"""
RAVEN ML Package

This package contains all machine learning components used by the
RAVEN Condition Parser system.

Components included:

- BoundaryDetector       : CRF model for detecting logical group boundaries
- OperatorClassifier     : TF-IDF + Logistic Regression classifier for operator phrases
- BM25Autocomplete       : BM25 based autocomplete engine for fields/operators
- confidence_gate        : compliance gate that validates model confidence

Utilities:

- train_all()            : training pipeline
- AuditLogger            : append-only ML audit logging

The helpers below make it easy to load models safely at server startup.
"""

from pathlib import Path
import pickle

from .models import (
    BoundaryDetector,
    OperatorClassifier,
    BM25Autocomplete,
    confidence_gate
)

from .trainer import train_all
from .audit import AuditLogger


# -------------------------------------------------------------
# Model loading helpers
# -------------------------------------------------------------

def load_boundary_detector(path: str = "models/boundary_detector.pkl"):
    """
    Safely load the CRF boundary detector.
    Returns None if file does not exist.
    """
    try:
        model = BoundaryDetector()
        return model.load(path)
    except Exception:
        return None


def load_operator_classifier(path: str = "models/operator_classifier.pkl"):
    """
    Safely load the operator classifier model.
    """
    try:
        model = OperatorClassifier()
        return model.load(path)
    except Exception:
        return None


def load_autocomplete(path: str = "models/autocomplete.pkl"):
    """
    Load BM25 autocomplete index.
    """
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


# -------------------------------------------------------------
# Public package API
# -------------------------------------------------------------

__all__ = [
    "BoundaryDetector",
    "OperatorClassifier",
    "BM25Autocomplete",
    "confidence_gate",
    "train_all",
    "AuditLogger",
    "load_boundary_detector",
    "load_operator_classifier",
    "load_autocomplete",
]
