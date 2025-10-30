# asap7_to_envjson.py
from collections import defaultdict
import json
import pathlib
import argparse
from typing import Dict, List, Tuple


def _aggregate_nets(rowN: List[dict], rowP: List[dict],
                    name_map_N: Dict[int, str],
                    name_map_P: Dict[int, str]) -> List[List[str]]:
    """
    把所有 pin 先依 netname 聚合，輸出為:
      [ "MN0.D", "MN1.S", ..., "<net_name>" ]
    最後一個元素是 net 名。
    """
    net2pins = defaultdict(list)

    def add(dev: str, pin: str, net: str):
        if net is None:
            return
        net2pins[net].append(f"{dev}.{pin}")

    # NMOS pins
    for i, t in enumerate(rowN):
        n = name_map_N[i]
        add(n, "D", t.get("net_drn", "N"))
        add(n, "G", t.get("gate_net", "G"))
        add(n, "S", t.get("net_src", "S"))
        add(n, "B", "VSS")  # bulk/body

    # PMOS pins
    for i, t in enumerate(rowP):
        p = name_map_P[i]
        add(p, "D", t.get("net_drn", "P"))
        add(p, "G", t.get("gate_net", "G"))
        add(p, "S", t.get("net_src", "S"))
        add(p, "B", "VDD")  # bulk/body

    nets = []
    for netname, pins in net2pins.items():
        pins_unique = []
        seen = set()
        # 去重，避免同一裝置在同一 net 重複被塞入
        for pin in pins:
            if pin not in seen:
                seen.add(pin)
                pins_unique.append(pin)
        if len(pins_unique) >= 1:  # 保留單 pin net 也行，下游會再過濾
            nets.append(pins_unique + [netname])
    return nets


def _build_pair_map(rowN: List[dict], rowP: List[dict],
                    name_map_N: Dict[int, str],
                    name_map_P: Dict[int, str]) -> Dict[str, str]:
    """
    以 gate_net 為主的 NMOS↔PMOS 智能配對：
      1) 先找 gate_net 完全相同的唯一對。
      2) 若多個 PMOS 候選，利用 (drain/source 是否共網) + W_norm 接近度打分。
      3) 還沒配到的，與剩餘對象順序補齊。
    """
    from math import inf

    # 依 gate_net 建索引
    p_by_gate = defaultdict(list)
    for j, tp in enumerate(rowP):
        g = tp.get("gate_net")
        if g is not None:
            p_by_gate[g].append(j)

    used_p = set()
    pair_map: Dict[str, str] = {}

    # 第一輪：唯一 gate 完全匹配
    for i, tn in enumerate(rowN):
        g = tn.get("gate_net")
        cand = [j for j in p_by_gate.get(g, []) if j not in used_p]
        if len(cand) == 1:
            j = cand[0]
            pair_map[name_map_N[i]] = name_map_P[j]
            pair_map[name_map_P[j]] = name_map_N[i]
            used_p.add(j)

    # 第二輪：多候選 → 打分挑最好
    def score_nmospmos(tn: dict, tp: dict) -> Tuple[int, float]:
        s = 0
        dn, sn = tn.get("net_drn"), tn.get("net_src")
        dp, sp = tp.get("net_drn"), tp.get("net_src")
        # drain/source 連到同 net 加分（與輸出/關鍵內部網越近越好）
        if dn is not None and (dn == dp or dn == sp):
            s += 2
        if sn is not None and (sn == dp or sn == sp):
            s += 1
        # W_norm 越接近越好（以負差做次序）
        wdiff = abs(float(tp.get("W_norm", 1.0)) -
                    float(tn.get("W_norm", 1.0)))
        return (s, -wdiff)

    for i, tn in enumerate(rowN):
        nname = name_map_N[i]
        if nname in pair_map:
            continue
        g = tn.get("gate_net")
        cand = [j for j in p_by_gate.get(g, []) if j not in used_p]
        if cand:
            best = (-1, -inf, None)
            for j in cand:
                sc = score_nmospmos(tn, rowP[j])
                if sc > best[:2]:
                    best = (sc[0], sc[1], j)
            j = best[2]
            if j is not None:
                pair_map[nname] = name_map_P[j]
                pair_map[name_map_P[j]] = nname
                used_p.add(j)

    # 第三輪：剩餘配對（保證雙向）
    remaining_p = [j for j in range(len(rowP)) if j not in used_p]
    idx = 0
    for i in range(len(rowN)):
        nname = name_map_N[i]
        if nname not in pair_map and idx < len(remaining_p):
            j = remaining_p[idx]
            idx += 1
            pair_map[nname] = name_map_P[j]
            pair_map[name_map_P[j]] = nname

    return pair_map


def _choose_pitch(prefix: dict, cell_name: str,
                  cli_poly: float = None, cli_row: float = None) -> Tuple[float, float]:
    """
    以資料集 prefix / family / CLI 得到 poly_pitch 與 row_pitch。
    ASAP7（7.5T）常用：
      poly_pitch = 0.054 μm（CPP 約 54 nm）
      row_pitch  = 0.270 μm（7.5 × M1_pitch，M1 pitch ≈ 0.036 μm）
    可由 CLI 覆寫。
    """
    # 1) dataset prefix
    poly = prefix.get("poly_pitch")
    cell_tracks = prefix.get("cell_height_tracks")

    # 2) family（用名稱判斷 7.5T）
    is_75t = ("_75t_" in cell_name) or ("_75T_" in cell_name)
    # 依公知 ASAP7 規格設定（可再細分不同家族）
    FAMILY_DEFAULT = {"poly_pitch": 0.054, "m1_pitch": 0.036, "tracks": 7.5}

    # 3) 決定 poly_pitch
    if poly is None and is_75t:
        poly = FAMILY_DEFAULT["poly_pitch"]
    if poly is None and cli_poly is not None:
        poly = cli_poly
    if poly is None:
        poly = FAMILY_DEFAULT["poly_pitch"]  # 最終保底

    # 4) 決定 row_pitch
    if cell_tracks is None and is_75t:
        cell_tracks = FAMILY_DEFAULT["tracks"]
    if cli_row is not None:
        row_pitch = cli_row
    else:
        # 若有 tracks 就用 tracks × M1_pitch；否則用 family 7.5T × M1_pitch
        m1p = FAMILY_DEFAULT["m1_pitch"]
        if cell_tracks is None:
            cell_tracks = FAMILY_DEFAULT["tracks"]
        row_pitch = float(cell_tracks) * m1p

    return float(poly), float(row_pitch)


def build_env_json(asap7_json: str, cell_name: str,
                   default_poly_pitch: float = None,
                   default_row_pitch: float = None) -> dict:
    data = json.load(open(asap7_json, "r"))
    cell = data[cell_name]

    rowP = cell["row_P"]
    rowN = cell["row_N"]

    # 1) devices
    devices = []
    name_map_P, name_map_N = {}, {}
    for i, t in enumerate(rowN):
        name = f"MN{i}"
        name_map_N[i] = name
        devices.append({
            "name": name, "type": "NMOS",
            "w": float(t.get("W_norm", 1.0)), "l": 0.05, "nf": 1, "vt": "SVT"
        })
    for i, t in enumerate(rowP):
        name = f"MP{i}"
        name_map_P[i] = name
        devices.append({
            "name": name, "type": "PMOS",
            "w": float(t.get("W_norm", 1.0)), "l": 0.05, "nf": 1, "vt": "SVT"
        })

    # 2) nets（聚合）
    nets = _aggregate_nets(rowN, rowP, name_map_N, name_map_P)

    # 3) constraints（智能配對 + pitch）
    pair_map = _build_pair_map(rowN, rowP, name_map_N, name_map_P)
    prefix = cell.get("prefix", {})

    # 允許 CLI 覆寫
    poly_pitch, row_pitch = _choose_pitch(
        prefix, cell_name, default_poly_pitch, default_row_pitch
    )

    constraints = {
        "pair_map": pair_map,
        "row_pitch": float(row_pitch),
        "poly_pitch": float(poly_pitch),
        "y_pmos": 1.0, "y_nmos": 0.0
    }

    return {"devices": devices, "nets": nets, "constraints": constraints}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="asap7_dataset.json 路徑")
    ap.add_argument("--cell", required=True, help="例如 AND2x2_ASAP7_75t_L")
    ap.add_argument("--out", required=True)
    ap.add_argument("--poly_pitch", type=float, default=None,
                    help="覆寫 poly pitch（μm），預設 ASAP7_75T=0.054")
    ap.add_argument("--row_pitch", type=float, default=None,
                    help="覆寫 row pitch（μm），預設 ASAP7_75T≈0.270")
    args = ap.parse_args()

    envj = build_env_json(args.dataset, args.cell,
                          args.poly_pitch, args.row_pitch)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(envj, open(args.out, "w"), indent=2)
    print(f"[OK] wrote {args.out}")
