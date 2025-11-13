import argparse
import random
import shutil
from pathlib import Path


def parse_split(s: str):
    a, b, c = map(int, s.split("/"))
    if a + b + c != 100:
        raise ValueError("Split must sum to 100, e.g., 80/10/10")
    return {"train": a, "val": b, "test": c}


def get_bins(root: Path):
    return [
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and "-" in p.name
        and all(x.isdigit() for x in p.name.split("-"))
    ]


def ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="preprocess/out_cells/_clean/bins")
    ap.add_argument("--target", default="preprocess/out_cells/_clean/bins")
    ap.add_argument("--bins", nargs="*", default=None)
    # default 改成 80/10/10
    ap.add_argument("--split", default="80/10/10")
    ap.add_argument(
        "--copy",
        action="store_true",
        help="Copy JSONs (otherwise only create folders)",
    )
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.source)
    tgt = Path(args.target)
    split = parse_split(args.split)
    bins = args.bins or get_bins(src)
    rng = random.Random(args.seed)

    print(
        f"Source: {src}\nTarget: {tgt}\nBins  : {bins}\nSplit : {split}  copy={args.copy} dry={args.dry_run}"
    )

    for b in bins:
        sbin = src / b
        if not sbin.exists():
            print(f"[SKIP] {b} (missing)")
            continue
        dbin = tgt / b
        for sub in ("train", "val", "test"):
            ensure(dbin / sub)

        jsons = sorted(sbin.glob("*.json"))
        if not jsons:
            print(f"[SKIP] {b} (no JSON)")
            continue

        rng.shuffle(jsons)
        n = len(jsons)
        n_train = round(n * split["train"] / 100)
        n_val = round(n * split["val"] / 100)
        n_test = n - n_train - n_val

        sets = {
            "train": jsons[:n_train],
            "val": jsons[n_train : n_train + n_val],
            "test": jsons[n_train + n_val :],
        }
        print(
            f"[{b}] JSON={n} → train={len(sets['train'])} / val={len(sets['val'])} / test={len(sets['test'])}"
        )

        if args.copy:
            for k, files in sets.items():
                dst = dbin / k
                for f in files:
                    target = dst / f.name
                    if target.exists():
                        continue
                    if not args.dry_run:
                        shutil.copy2(f, target)

    print("Done.")


if __name__ == "__main__":
    main()
