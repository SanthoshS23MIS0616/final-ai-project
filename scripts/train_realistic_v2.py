from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.training_realistic_v2 import train_and_save


def main() -> None:
    print(json.dumps(train_and_save(), indent=2))


if __name__ == "__main__":
    main()
