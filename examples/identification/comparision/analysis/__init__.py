"""
The analyses the identification routes.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Reached only by `python -m analysis.<name>`, where sys.path holds comparision but not this folder
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
