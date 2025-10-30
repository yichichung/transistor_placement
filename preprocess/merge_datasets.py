import json
import sys
from pathlib import Path


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python merge_datasets.py <out.json> <in1.json> [in2.json ...]")
        sys.exit(1)

    out_path = Path(sys.argv[1])
    merged = {}

    for p in sys.argv[2:]:
        data = load(p)
        # 支援兩種結構：{cell:{...}} 或 {"cells":{cell:{...}}}
        if "cells" in data and isinstance(data["cells"], dict):
            data = data["cells"]
        for k, v in data.items():
            merged[k] = v

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[OK] merged {len(merged)} cells -> {out_path}")


if __name__ == "__main__":
    main()
