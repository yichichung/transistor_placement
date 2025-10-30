# -*- coding: utf-8 -*-
# preprocess/cdl_to_dataset.py
import re
import os
import json
import argparse
import glob
from collections import defaultdict

SUBCKT_RE = re.compile(r"^\s*\.subckt\s+(\S+)\s+(.*)$", re.IGNORECASE)
END_RE = re.compile(r"^\s*\.ends", re.IGNORECASE)
MOS_RE = re.compile(
    r"^\s*[Mm]\S*\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", re.IGNORECASE)
#           Mname  D      G      S      B      model   [params...]


def parse_cdl_file(path):
    cells = {}
    cur = None
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("*"):  # skip comments
                continue
            m = SUBCKT_RE.match(line)
            if m:
                cur = {"name": m.group(1), "pins": m.group(
                    2).split(), "mos": []}
                cells[cur["name"]] = cur
                continue
            if cur and END_RE.match(line):
                cur = None
                continue
            if cur:
                mm = MOS_RE.match(line)
                if mm:
                    # D G S B model
                    d, g, s, b, model = mm.groups()
                    # 判別 NMOS/PMOS：ASAP7 多用 nmos/pmos 模型名，簡化以 p/P 開頭視為 PMOS
                    typ = "PMOS" if model.lower().startswith("p") else "NMOS"
                    cur["mos"].append({"name": mm.group(0).split()[
                                      0], "d": d, "g": g, "s": s, "b": b, "type": typ})
    return cells


def scan_inputs(path):
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.cdl")) + \
            glob.glob(os.path.join(path, "*.sp"))
    else:
        files = [path]
    out = {}
    for fp in files:
        out.update(parse_cdl_file(fp))
    return out


def build_dataset(cells):
    dataset = {}
    for cell_name, c in cells.items():
        row_P, row_N = [], []
        # 先把 MOS 依出現順序分到 row_N / row_P
        for i, m in enumerate(c["mos"]):
            rec = {
                "W_norm": 1.0,                # 先給個占位/可依參數抽取
                "poly_parity": (i % 2),       # 示意特徵，可日後改真值
                "net_src": m["s"],
                "net_drn": m["d"],
                "gate_net": m["g"]
            }
            if m["type"] == "PMOS":
                row_P.append(rec)
            else:
                row_N.append(rec)
        dataset[cell_name] = {
            "prefix": {
                "poly_pitch": None,
                "od_min_space": None,
                "cell_height_tracks": None,
                "poly_dir": 1
            },
            "row_P": row_P,
            "row_N": row_N
        }
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cdl", help="CDL file or folder")
    ap.add_argument("--sp", help="SPICE file or folder (alternative)")
    ap.add_argument("--out", required=True, help="output dataset json")
    args = ap.parse_args()

    src = args.cdl or args.sp
    if not src:
        raise SystemExit("Please provide --cdl or --sp")

    cells = scan_inputs(src)
    if not cells:
        raise SystemExit("No cells found in given input")

    ds = build_dataset(cells)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ds, f, indent=2)
    print(f"[OK] wrote {args.out} with {len(ds)} cells")


if __name__ == "__main__":
    main()
