import sys
import os

# Make the project root importable so `import parser` resolves our local package
# (not the stdlib parser module) by inserting it at position 0 in sys.path.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from parser import parse_condition, LexerError, ParseError
from ML import load_operator_classifier
from ML.models import BM25Autocomplete
from ML.data import BM25_CORPUS


# ------------------------------------------------------------------ #
# App setup
# ------------------------------------------------------------------ #

app = FastAPI(title="RAVEN Condition Parser", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

operator_model = load_operator_classifier("models/operator_classifier.pkl")
if operator_model is None:
    print("[main] OperatorClassifier not loaded — run ML/train.py first")

autocomplete_model: Optional[BM25Autocomplete] = None
try:
    _ac = BM25Autocomplete()
    _ac.fit(BM25_CORPUS)
    autocomplete_model = _ac
    print("[main] BM25Autocomplete ready")
except Exception as e:
    print(f"[main] BM25Autocomplete not available: {e}")


# ------------------------------------------------------------------ #
# Request / Response Models
# ------------------------------------------------------------------ #

class ParseRequest(BaseModel):
    text: str

class OperatorRequest(BaseModel):
    text: str

class ValidateRequest(BaseModel):
    text: str


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #

@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/api/parse")
def parse(req: ParseRequest):
    try:
        result = parse_condition(req.text)
        return {"success": True, "data": result}
    except (LexerError, ParseError) as e:
        return {
            "success": False,
            "error":   str(e),
            "position": getattr(e, "position", -1),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "position": -1}


@app.post("/api/validate")
def validate(req: ValidateRequest):
    """
    Validate a condition string without returning the full AST.
    Returns {valid: true} or {valid: false, error: "...", position: N}.
    """
    try:
        parse_condition(req.text)
        return {"valid": True}
    except (LexerError, ParseError) as e:
        return {"valid": False, "error": str(e), "position": getattr(e, "position", -1)}
    except Exception as e:
        return {"valid": False, "error": str(e), "position": -1}


@app.post("/api/operator")
def operator(req: OperatorRequest):
    if not operator_model:
        raise HTTPException(status_code=503, detail="Operator model not loaded")
    return operator_model.predict(req.text)


@app.get("/api/autocomplete")
def autocomplete(q: str = "", k: int = 5):
    if not autocomplete_model:
        return {"results": []}
    return {"results": autocomplete_model.query(q, top_k=k)}