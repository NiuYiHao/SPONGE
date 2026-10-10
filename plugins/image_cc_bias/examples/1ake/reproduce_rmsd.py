"""Redraw the 45 held-out 1AKE RMSD pairs and highlight target 19."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _rmsd import main

if __name__ == "__main__":
    main("1AKE", selected_target=19)
