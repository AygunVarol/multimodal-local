"""Compatibility wrapper for the renamed safety-experiment runner."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_safety_experiment import main  # noqa: E402


if __name__ == "__main__":
    main()
