<<<<<<< HEAD
# dedup_and_bin_cells.py
import json
import hashlib
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

# —— 你只需要改這個根目錄 —— #
DEFAULT_ROOT = Path(
    r"C:\Users\USER\Downloads\transistor_placement_pkg\preprocess\out_cells")

# 分桶規則（含上界）
BUCKETS = [
    (0, 5), (6, 9), (10, 15), (16, 20), (21, 25), (26,
                                                   30), (31, 50), (51, 100), (101, 200), (201, 10_000)
]
PREF_ORDER = ["_L", "_R", "_SL"]  # 保留優先序（名稱包含此子字串者優先）


def bucket_of(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "unbinned"


def json_bytes_normalized(d: Dict) -> bytes:
    # 嚴格內容相等用；鍵排序 + 移除空白
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def device_tuple(dev: Dict) -> Tuple:
    # 結構相等用：type/w/l/nf/vt
    return (
        str(dev.get("type", "")).upper(),
        float(dev.get("w", 0.0)),
        float(dev.get("l", 0.0)),
        int(dev.get("nf", 1)),
        str(dev.get("vt", "")).upper()
    )


def canonical_struct(d: Dict) -> Tuple:
    # 結構指紋：devices（排序） + nets（每條 net 的元素排序，整體再排序）
    devs = d.get("devices", [])
    nets = d.get("nets", [])
    dev_key = tuple(sorted(device_tuple(x) for x in devs))
    # 每條 net 內部排序；整體也排序，避免順序差異
    nets_key = tuple(sorted(tuple(sorted(map(str, net))) for net in nets))
    return (dev_key, nets_key)


def family_key(p: Path) -> str:
    # 將常見 corner 後綴（_L/_R/_SL/_SRAM 等）視為同一「家族」
    stem = p.stem
    for tag in ["_L", "_R", "_SL", "_SR", "_SRAM"]:
        if stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def pref_rank(name: str) -> int:
    for i, k in enumerate(PREF_ORDER):
        if k in name:
            return i
    return len(PREF_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="folder of JSON cells")
    ap.add_argument("--apply", action="store_true",
                    help="really move files (otherwise dry-run)")
    args = ap.parse_args()

    root: Path = args.root
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"[!] No json under {root}")
        return

    # 掃描並建立兩種指紋
    exact_map: Dict[str, List[Path]] = {}
    struct_map: Dict[Tuple, List[Path]] = {}

    parsed: Dict[Path, Dict] = {}
    mos_count: Dict[Path, int] = {}

    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        parsed[p] = d
        mos = len(d.get("devices", []))
        mos_count[p] = mos

        h_exact = hashlib.sha256(json_bytes_normalized(d)).hexdigest()
        exact_map.setdefault(h_exact, []).append(p)

        key_struct = canonical_struct(d)
        struct_map.setdefault(key_struct, []).append(p)

    # 先找 exact duplicates
    exact_dups: List[List[Path]] = [
        grp for grp in exact_map.values() if len(grp) > 1]

    # 再找 structural duplicates（把 exact 已經抓到的排除）
    seen_exact = set()
    for grp in exact_dups:
        seen_exact.update(grp)

    struct_dups: List[List[Path]] = []
    for grp in struct_map.values():
        group = [p for p in grp if p not in seen_exact]
        if len(group) > 1:
            struct_dups.append(group)

    # 依家族優先序決定保留與刪除
    keep = set()
    drop = set()

    def choose_keep(group: List[Path]) -> Path:
        # 以 family 分簇後各自挑一個；最後把所有候選再比一次優先序
        # 簡化處理：整組直接挑優先序最前 + 路徑最短 + 檔名最短
        return sorted(group, key=lambda p: (pref_rank(p.name), len(p.name), len(str(p))))[0]

    for grp in exact_dups + struct_dups:
        k = choose_keep(grp)
        keep.add(k)
        for p in grp:
            if p != k:
                drop.add(p)

    # 其他未命中的全列入 keep
    for p in files:
        if p not in drop:
            keep.add(p)

    print("\n=== DRY RUN ===" if not args.apply else "\n=== APPLY MODE ===")
    print(f"Total files: {len(files)}")
    print(f"Keep: {len(keep)}  |  Drop(duplicates): {len(drop)}")

    # 列出去重對照
    report_lines = ["action,file,mos,bucket"]
    for p in sorted(keep, key=lambda x: x.name):
        report_lines.append(
            f"KEEP,{p},{mos_count.get(p, '')},{bucket_of(mos_count.get(p, 0))}")
    for p in sorted(drop, key=lambda x: x.name):
        report_lines.append(
            f"DROP,{p},{mos_count.get(p, '')},{bucket_of(mos_count.get(p, 0))}")

    out_clean = root / "_clean"
    out_dups = root / "_duplicates"
    out_clean.mkdir(exist_ok=True)
    out_dups.mkdir(exist_ok=True)

    (root / "dedup_report.csv").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[report] {root/'dedup_report.csv'}")

    if not args.apply:
        print(
            "\n[Dry-run only] No files moved. Re-run with --apply to apply changes.")
        return

    # 實際搬檔
    for p in keep:
        # 分桶資料夾
        b = bucket_of(mos_count.get(p, 0))
        dst_dir = out_clean / "bins" / b
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst_dir / p.name))

    for p in drop:
        shutil.move(str(p), str(out_dups / p.name))

    # 產生分桶摘要
    from collections import Counter
    cnt = Counter(bucket_of(mos_count.get(p, 0)) for p in keep)
    lines = ["bucket,num_files"]
    for lo, hi in BUCKETS:
        lab = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        lines.append(f"{lab},{cnt.get(lab, 0)}")
    (root / "bin_summary.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] moved kept files into {out_clean/'bins'}")
    print(f"[done] duplicates moved into {out_dups}")
    print(f"[summary] {root/'bin_summary.csv'}")
    print("\nTip: 之後訓練可直接用  e.g.  --env-dir  ...\\out_cells\\_clean\\bins\\10-15")


if __name__ == "__main__":
    main()
=======
# dedup_and_bin_cells.py
import json
import hashlib
import argparse
import shutil
from pathlib import Path
from typing import Dict, Tuple, List

# —— 你只需要改這個根目錄 —— #
DEFAULT_ROOT = Path(
    r"C:\Users\USER\Downloads\transistor_placement_pkg\preprocess\out_cells")

# 分桶規則（含上界）
BUCKETS = [
    (0, 5), (6, 9), (10, 15), (16, 20), (21, 25), (26,
                                                   30), (31, 50), (51, 100), (101, 200), (201, 10_000)
]
PREF_ORDER = ["_L", "_R", "_SL"]  # 保留優先序（名稱包含此子字串者優先）


def bucket_of(n: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
    return "unbinned"


def json_bytes_normalized(d: Dict) -> bytes:
    # 嚴格內容相等用；鍵排序 + 移除空白
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


def device_tuple(dev: Dict) -> Tuple:
    # 結構相等用：type/w/l/nf/vt
    return (
        str(dev.get("type", "")).upper(),
        float(dev.get("w", 0.0)),
        float(dev.get("l", 0.0)),
        int(dev.get("nf", 1)),
        str(dev.get("vt", "")).upper()
    )


def canonical_struct(d: Dict) -> Tuple:
    # 結構指紋：devices（排序） + nets（每條 net 的元素排序，整體再排序）
    devs = d.get("devices", [])
    nets = d.get("nets", [])
    dev_key = tuple(sorted(device_tuple(x) for x in devs))
    # 每條 net 內部排序；整體也排序，避免順序差異
    nets_key = tuple(sorted(tuple(sorted(map(str, net))) for net in nets))
    return (dev_key, nets_key)


def family_key(p: Path) -> str:
    # 將常見 corner 後綴（_L/_R/_SL/_SRAM 等）視為同一「家族」
    stem = p.stem
    for tag in ["_L", "_R", "_SL", "_SR", "_SRAM"]:
        if stem.endswith(tag):
            return stem[: -len(tag)]
    return stem


def pref_rank(name: str) -> int:
    for i, k in enumerate(PREF_ORDER):
        if k in name:
            return i
    return len(PREF_ORDER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="folder of JSON cells")
    ap.add_argument("--apply", action="store_true",
                    help="really move files (otherwise dry-run)")
    args = ap.parse_args()

    root: Path = args.root
    files = sorted(root.rglob("*.json"))
    if not files:
        print(f"[!] No json under {root}")
        return

    # 掃描並建立兩種指紋
    exact_map: Dict[str, List[Path]] = {}
    struct_map: Dict[Tuple, List[Path]] = {}

    parsed: Dict[Path, Dict] = {}
    mos_count: Dict[Path, int] = {}

    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue
        parsed[p] = d
        mos = len(d.get("devices", []))
        mos_count[p] = mos

        h_exact = hashlib.sha256(json_bytes_normalized(d)).hexdigest()
        exact_map.setdefault(h_exact, []).append(p)

        key_struct = canonical_struct(d)
        struct_map.setdefault(key_struct, []).append(p)

    # 先找 exact duplicates
    exact_dups: List[List[Path]] = [
        grp for grp in exact_map.values() if len(grp) > 1]

    # 再找 structural duplicates（把 exact 已經抓到的排除）
    seen_exact = set()
    for grp in exact_dups:
        seen_exact.update(grp)

    struct_dups: List[List[Path]] = []
    for grp in struct_map.values():
        group = [p for p in grp if p not in seen_exact]
        if len(group) > 1:
            struct_dups.append(group)

    # 依家族優先序決定保留與刪除
    keep = set()
    drop = set()

    def choose_keep(group: List[Path]) -> Path:
        # 以 family 分簇後各自挑一個；最後把所有候選再比一次優先序
        # 簡化處理：整組直接挑優先序最前 + 路徑最短 + 檔名最短
        return sorted(group, key=lambda p: (pref_rank(p.name), len(p.name), len(str(p))))[0]

    for grp in exact_dups + struct_dups:
        k = choose_keep(grp)
        keep.add(k)
        for p in grp:
            if p != k:
                drop.add(p)

    # 其他未命中的全列入 keep
    for p in files:
        if p not in drop:
            keep.add(p)

    print("\n=== DRY RUN ===" if not args.apply else "\n=== APPLY MODE ===")
    print(f"Total files: {len(files)}")
    print(f"Keep: {len(keep)}  |  Drop(duplicates): {len(drop)}")

    # 列出去重對照
    report_lines = ["action,file,mos,bucket"]
    for p in sorted(keep, key=lambda x: x.name):
        report_lines.append(
            f"KEEP,{p},{mos_count.get(p, '')},{bucket_of(mos_count.get(p, 0))}")
    for p in sorted(drop, key=lambda x: x.name):
        report_lines.append(
            f"DROP,{p},{mos_count.get(p, '')},{bucket_of(mos_count.get(p, 0))}")

    out_clean = root / "_clean"
    out_dups = root / "_duplicates"
    out_clean.mkdir(exist_ok=True)
    out_dups.mkdir(exist_ok=True)

    (root / "dedup_report.csv").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[report] {root/'dedup_report.csv'}")

    if not args.apply:
        print(
            "\n[Dry-run only] No files moved. Re-run with --apply to apply changes.")
        return

    # 實際搬檔
    for p in keep:
        # 分桶資料夾
        b = bucket_of(mos_count.get(p, 0))
        dst_dir = out_clean / "bins" / b
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dst_dir / p.name))

    for p in drop:
        shutil.move(str(p), str(out_dups / p.name))

    # 產生分桶摘要
    from collections import Counter
    cnt = Counter(bucket_of(mos_count.get(p, 0)) for p in keep)
    lines = ["bucket,num_files"]
    for lo, hi in BUCKETS:
        lab = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        lines.append(f"{lab},{cnt.get(lab, 0)}")
    (root / "bin_summary.csv").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] moved kept files into {out_clean/'bins'}")
    print(f"[done] duplicates moved into {out_dups}")
    print(f"[summary] {root/'bin_summary.csv'}")
    print("\nTip: 之後訓練可直接用  e.g.  --env-dir  ...\\out_cells\\_clean\\bins\\10-15")


if __name__ == "__main__":
    main()
>>>>>>> b11876e6fcbff28a3f5b2c6856ed503f70b7bc9d
