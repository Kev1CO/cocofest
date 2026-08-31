from __future__ import annotations

import sys
from pathlib import Path


sys.dont_write_bytecode = True

PROCEDURE_ROOT = Path(__file__).resolve().parent
COMPARISON_ROOT = PROCEDURE_ROOT.parent
for _root in (PROCEDURE_ROOT, COMPARISON_ROOT, COMPARISON_ROOT / "optimization"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from helper import debug_plots, results


RESULTS_ROOT = results.RESULTS_ROOT
DEBUG_ROOT = debug_plots.DEBUG_ROOT
DATA_ROOT = COMPARISON_ROOT / "data"
MODEL_ROOT = COMPARISON_ROOT / "model"


def step_folders(method) -> dict:
    """Where the per step figures of one route go, one folder per step of the article's algorithm."""
    return {f"step{i}": DEBUG_ROOT / method / "steps" / f"step{i}" for i in range(1, 6)}
