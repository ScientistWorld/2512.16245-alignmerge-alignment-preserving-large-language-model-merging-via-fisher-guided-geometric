import argparse
import json
from pathlib import Path


def validate(path: Path):
    data = json.loads(path.read_text())
    assert "experiments" in data
    for exp_name, exp in data["experiments"].items():
        assert "results" in exp, exp_name
        assert exp["results"], exp_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    validate(Path("scoring/reference.json"))
    if Path("scoring/scores.json").exists():
        validate(Path("scoring/scores.json"))
    if args.compare and Path("scoring/scores.json").exists():
        ref = json.loads(Path("scoring/reference.json").read_text())
        scores = json.loads(Path("scoring/scores.json").read_text())
        print("Reference experiments:", ", ".join(ref["experiments"]))
        print(json.dumps(scores, indent=2))
    print("validation ok")


if __name__ == "__main__":
    main()
