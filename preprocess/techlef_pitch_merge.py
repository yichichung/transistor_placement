# -*- coding: utf-8 -*-
# preprocess/techlef_pitch_merge.py
import re
import json
import argparse
import os

PITCH_RE = re.compile(
    r"^ *SITE +(\S+) *;|^ *SIZE +([\d.]+) BY +([\d.]+) *;", re.IGNORECASE)
# 簡化：實務上可讀 SITE/ROW/UNITS 等，這裡保守抓，抓不到就不覆蓋


def extract_pitch(lef_path):
    poly_pitch = None
    row_pitch = None
    with open(lef_path, "r") as f:
        for line in f:
            m = PITCH_RE.match(line)
            if not m:
                continue
            # 這個簡化版僅示意；實務你可以用更精準的 parser 讀 GRID/PITCH
    return poly_pitch, row_pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lef", required=True)
    ap.add_argument("--json-in", required=True)
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.json_in, "r"))
    poly, row = extract_pitch(args.lef)

    # 讀不到 pitch 就維持原值；你也可以在這裡塞默認 0.056 / 1.0
    for cell_name, cell in data.items():
        pref = cell.setdefault("prefix", {})
        if poly is not None:
            pref["poly_pitch"] = poly
        if row is not None:
            pref["row_pitch"] = row

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] merged pitch into {args.json_out}")


if __name__ == "__main__":
    main()
