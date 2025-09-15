# log/logparser/logparser/utils.py
"""
Minimal stub utils for the bundled Drain parser.

This is intentionally small: it supplies an `evaluator` object and a
couple of lightweight helpers so imports succeed. It does NOT attempt
to reproduce full LogPAI functionality (evaluation metrics, plots, etc).
If later code requires additional functions, we'll extend this file.
"""

from typing import Any, Dict, List

class _DummyEvaluator:
    """Lightweight evaluator stub — add more methods later if needed."""
    def __init__(self):
        self.name = "dummy-evaluator"

    def evaluate(self, *args, **kwargs) -> Any:
        """Generic placeholder. Returns None / empty results."""
        return None

    def score(self, *args, **kwargs) -> Dict:
        """Return an empty score dict placeholder."""
        return {}

# Expose instance named `evaluator` so `from .utils import evaluator` works.
evaluator = _DummyEvaluator()

# Small helper utilities that other code sometimes expects.
def ensure_dir(path: str) -> None:
    """Create directory if missing (no error if exists)."""
    import os
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Any) -> None:
    """Simple JSON dump helper."""
    import json, io
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str) -> Any:
    """Simple JSON load helper."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
