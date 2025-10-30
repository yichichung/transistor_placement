import json
from pathlib import Path
import argparse
from typing import Dict, Any


def normalize_dataset(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    接受兩種格式：
      A) { "<cell_name>": {...}, ... }
      B) { "cells": { "<cell_name>": {...}, ... } }
    統一回傳格式 A。
    """
    if isinstance(obj, dict) and "cells" in obj and isinstance(obj["cells"], dict):
        return obj["cells"]
    return obj


def count_devices(cell: Dict[str, Any]) -> int:
    n = 0
    for k in ("row_P", "row_N"):
        if k in cell and isinstance(cell[k], list):
            n += len(cell[k])
    return n


def to_envjson(cell_name: str, cell: Dict[str, Any], poly_pitch: float, row_pitch: float) -> Dict[str, Any]:
    """
    將 dataset 的一筆 cell 轉為環境 JSON（與 asap7_to_envjson.py 同構）。
    """
    # nets 直接沿用（若不存在則給空陣列）
    nets = cell.get("nets", [])

    # 產生 pair_map（若已有就沿用）
    pair_map = {}
    if "pair_map" in cell and isinstance(cell["pair_map"], dict):
        pair_map = cell["pair_map"]
    else:
        # 依序配對：第 i 個 nmos 配第 i 個 pmos（數量不等則配到最小值）
        nmos = [d["name"] for d in cell.get("row_N", []) if d.get(
            "type", "").lower().startswith("n")]
        pmos = [d["name"] for d in cell.get("row_P", []) if d.get(
            "type", "").lower().startswith("p")]
        m = min(len(nmos), len(pmos))
        for i in range(m):
            pair_map[nmos[i]] = pmos[i]
            pair_map[pmos[i]] = nmos[i]

    # y-row 座標（沿用既有或預設）
    constraints = {
        "pair_map": pair_map,
        "row_pitch": row_pitch,
        "poly_pitch": poly_pitch,
        "y_pmos": cell.get("constraints", {}).get("y_pmos", 1.0),
        "y_nmos": cell.get("constraints", {}).get("y_nmos", 0.0)
    }

    # 組 env json
    env = {
        "cell_name": cell_name,
        "devices": cell.get("row_P", []) + cell.get("row_N", []),
        "nets": nets,
        "constraints": constraints
    }
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--poly_pitch", type=float, required=True)
    ap.add_argument("--row_pitch", type=float, required=True)
    ap.add_argument("--min_devices", type=int, default=3,
                    help="小於此器件數的 cell 會被跳過")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    cells = normalize_dataset(data)
    exported = 0
    skipped = 0

    for name, cell in cells.items():
        nd = count_devices(cell)
        if nd < args.min_devices:
            print(
                f"[SKIP] {name}: devices={nd} < min_devices={args.min_devices}")
            skipped += 1
            continue
        env = to_envjson(name, cell, args.poly_pitch, args.row_pitch)
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)
        exported += 1

    print(
        f"[OK] exported {exported} env jsons to {out_dir} (min_devices={args.min_devices}, skipped={skipped})")


if __name__ == "__main__":
    main()
