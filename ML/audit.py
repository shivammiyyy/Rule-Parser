"""
AuditLogger — append-only ML audit logging.

Each prediction event is written as a single JSON line (JSONL format)
to a log file, making it easy to stream, grep, or load into pandas.

Usage
-----
    logger = AuditLogger("logs/ml_audit.jsonl")
    logger.log("operator_classifier", phrase="lower than",
               prediction="lessThan", confidence=0.97)
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


class AuditLogger:
    """
    Thread-safe, append-only logger for ML prediction events.

    Each call to log() appends one JSON object on a new line:

        {
          "timestamp": "2026-03-16T18:10:00+00:00",
          "model":     "operator_classifier",
          "phrase":    "lower than",
          "prediction":"lessThan",
          "confidence":0.97,
          ...
        }
    """

    def __init__(self, path: str = "logs/ml_audit.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, model: str, **kwargs) -> dict:
        """
        Append one audit record to the log file.

        Parameters
        ----------
        model  : logical name of the model (e.g. "operator_classifier")
        kwargs : arbitrary key-value pairs to include in the record
                 (e.g. phrase=..., prediction=..., confidence=...)

        Returns
        -------
        The dict that was written (useful for unit-testing).
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            **kwargs,
        }

        line = json.dumps(record, default=str)

        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        return record

    # ------------------------------------------------------------------
    # Read-back helpers (non-mutating)
    # ------------------------------------------------------------------

    def read_all(self) -> list[dict]:
        """Return all records as a list of dicts (newest last)."""
        if not self._path.exists():
            return []
        records = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def tail(self, n: int = 20) -> list[dict]:
        """Return the last n records."""
        return self.read_all()[-n:]

    def __repr__(self) -> str:
        return f"AuditLogger(path={self._path!r})"
