from pathlib import Path
import json

from src.skin_optics_so_r1.realface_classifier_input_preparation import run


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(json.dumps(run(root), indent=2, sort_keys=True))

