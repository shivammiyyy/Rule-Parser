"""
ML models for the RAVEN Condition Parser.

  BoundaryDetector   — CRF model for detecting logical group boundaries
  OperatorClassifier — TF-IDF + Logistic Regression for operator phrases
  BM25Autocomplete   — BM25-based autocomplete for fields / operators
  confidence_gate    — compliance gate that validates model confidence
"""

import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    import sklearn_crfsuite
    _CRF_AVAILABLE = True
except ImportError:
    _CRF_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


# ------------------------------------------------------------------ #
# BoundaryDetector                                                     #
# ------------------------------------------------------------------ #

class BoundaryDetector:
    """
    CRF model that labels tokens with B-GROUP / I-GROUP / O to detect
    logical group boundaries in a condition string.

    Requires:  sklearn-crfsuite
    """

    def __init__(self, algorithm: str = "lbfgs", c1: float = 0.1, c2: float = 0.1,
                 max_iterations: int = 100):

        if not _CRF_AVAILABLE:
            raise ImportError(
                "sklearn-crfsuite is required for BoundaryDetector. "
                "Install it with: pip install sklearn-crfsuite"
            )

        self.crf = sklearn_crfsuite.CRF( # type: ignore
            algorithm=algorithm,
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _token_features(tokens: list[str], idx: int) -> dict:
        """Build a feature dict for the token at position idx."""
        word = tokens[idx]

        feats = {
            "bias": 1.0,
            "word.lower": word.lower(),
            "word.isupper": word.isupper(),
            "word.isdigit": word.isdigit(),
            "word[:2]": word[:2],
            "word[:3]": word[:3],
        }

        if idx > 0:
            prev = tokens[idx - 1]
            feats["prev.lower"] = prev.lower()
            feats["prev.isupper"] = prev.isupper()
        else:
            feats["BOS"] = True  # Beginning Of Sentence

        if idx < len(tokens) - 1:
            nxt = tokens[idx + 1]
            feats["next.lower"] = nxt.lower()
            feats["next.isupper"] = nxt.isupper()
        else:
            feats["EOS"] = True  # End Of Sentence

        return feats

    def _sequence_features(self, tokens: list[str]) -> list[dict]:
        return [self._token_features(tokens, i) for i in range(len(tokens))]

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, sentences: list[list[str]], labels: list[list[str]]):
        """
        Train on a list of token sequences and their BIO label sequences.

        sentences : [["age", ">", "30", "AND", "status", "=", "active"], ...]
        labels    : [["O", "O", "O", "O", "B-GROUP", "O", "I-GROUP"], ...]
        """
        X = [self._sequence_features(s) for s in sentences]
        self.crf.fit(X, labels)
        return self

    def predict(self, tokens: list[str]) -> list[str]:
        """Return BIO labels for a single token sequence."""
        X = [self._sequence_features(tokens)]
        return self.crf.predict(X)[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/boundary_detector.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.crf, f)
        return self

    def load(self, path: str = "models/boundary_detector.pkl"):
        with open(path, "rb") as f:
            self.crf = pickle.load(f)
        return self


# ------------------------------------------------------------------ #
# OperatorClassifier                                                   #
# ------------------------------------------------------------------ #

class OperatorClassifier:
    """
    TF-IDF + Logistic Regression classifier that maps natural-language
    operator phrases (e.g. "lower than") to canonical operators
    (e.g. "lessThan").
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), analyzer="word")),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ])

    def fit(self, samples: list[tuple[str, str]]):
        """
        samples: [(phrase, label), ...]
            e.g. [("lower than", "lessThan"), ("above", "greaterThan"), ...]
        """
        X = [s[0] for s in samples]
        y = [s[1] for s in samples]
        self.pipeline.fit(X, y)
        return self

    def predict(self, phrase: str) -> dict:
        """Return prediction and confidence score."""
        probs = self.pipeline.predict_proba([phrase])[0]
        pred = self.pipeline.predict([phrase])[0]
        return {
            "prediction": pred,
            "confidence": float(max(probs)),
        }

    def save(self, path: str = "models/operator_classifier.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        return self

    def load(self, path: str = "models/operator_classifier.pkl"):
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        return self


# ------------------------------------------------------------------ #
# BM25Autocomplete                                                     #
# ------------------------------------------------------------------ #

class BM25Autocomplete:
    """
    BM25-based autocomplete engine for field names and operator phrases.

    Requires:  rank_bm25
    """

    def __init__(self):
        if not _BM25_AVAILABLE:
            raise ImportError(
                "rank_bm25 is required for BM25Autocomplete. "
                "Install it with: pip install rank_bm25"
            )
        self._corpus: list[str] = []
        self._bm25 = None

    def fit(self, corpus: list[str]):
        """
        Index a list of field / operator strings for autocomplete.

        corpus: ["accountBalance", "status", "greaterThan", "lessThan", ...]
        """
        self._corpus = corpus
        tokenized = [doc.lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized) # type: ignore
        return self

    def query(self, prefix: str, top_k: int = 5) -> list[str]:
        """
        Return the top-k corpus entries that best match the given prefix.
        """
        if self._bm25 is None:
            return []

        # Prefix filter first — fast path for exact starts
        exact = [c for c in self._corpus if c.lower().startswith(prefix.lower())]
        if exact:
            return exact[:top_k]

        # Fall back to BM25 ranking
        tokens = prefix.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self._corpus[i] for i in ranked[:top_k] if scores[i] > 0]

    def save(self, path: str = "models/autocomplete.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return self

    def load(self, path: str = "models/autocomplete.pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self._corpus = obj._corpus
        self._bm25 = obj._bm25
        return self


# ------------------------------------------------------------------ #
# confidence_gate                                                      #
# ------------------------------------------------------------------ #

class ConfidenceBelowThreshold(Exception):
    """Raised by confidence_gate when prediction confidence is too low."""


def confidence_gate(result: dict, threshold: float = 0.75) -> dict:
    """
    Validate that a model's prediction confidence meets the required threshold.

    Parameters
    ----------
    result    : dict returned by model.predict() — must contain "confidence" key.
    threshold : minimum acceptable confidence (default 0.75).

    Returns
    -------
    The same result dict if confidence >= threshold.

    Raises
    ------
    ConfidenceBelowThreshold if confidence < threshold.
    """
    confidence = result.get("confidence", 0.0)

    if confidence < threshold:
        raise ConfidenceBelowThreshold(
            f"Prediction confidence {confidence:.3f} is below threshold {threshold:.3f}. "
            f"Prediction: {result.get('prediction', 'unknown')}"
        )

    return result