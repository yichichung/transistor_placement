# preprocess/batch_make_envjson.py
import json
import argparse
import subprocess
import pathlib
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--poly_pitch", type=float, default=0.054)
    ap.add_argument("--row_pitch",  type=float, default=0.270)
    ap.add_argument("--include", nargs="*", default=[], help="只轉這些正則/關鍵字（OR）")
    ap.add_argument("--exclude", nargs="*", default=[], help="排除這些正則/關鍵字（OR）")
    args = ap.parse_args()

    ds = json.loads(pathlib.Path(args.dataset).read_text())
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def keep(name: str) -> bool:
        if args.include:
            if not any(re.search(pat, name) for pat in args.include):
                return False
        if args.exclude:
            if any(re.search(pat, name) for pat in args.exclude):
                return False
        return True

    cells = [c for c in ds.keys() if keep(c)]
    print(f"[INFO] total cells in dataset: {len(ds)}; selected: {len(cells)}")
    for c in sorted(cells):
        out = outdir / f"{c}.json"
        cmd = [
            "python", "preprocess/asap7_to_envjson.py",
            "--dataset", args.dataset,
            "--cell", c,
            "--out", str(out),
            "--poly_pitch", str(args.poly_pitch),
            "--row_pitch",  str(args.row_pitch),
        ]
        print("[RUN]", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
