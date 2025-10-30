import re
import json
from collections import defaultdict

def parse_spice(filename):
    cells = {}
    with open(filename, "r") as f:
        lines = f.readlines()

    subckt_name = None
    current_transistors = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("*"):  # skip comment
            continue

        # subckt start
        if line.lower().startswith(".subckt"):
            parts = line.split()
            subckt_name = parts[1]
            current_transistors = []
            continue

        # subckt end
        if line.lower().startswith(".ends"):
            if subckt_name and current_transistors:
                cells[subckt_name] = current_transistors
            subckt_name = None
            continue

        # MOSFET line
        if subckt_name and line.startswith("M"):
            parts = line.split()
            name = parts[0]
            drain, gate, source, bulk = parts[1:5]
            model = parts[5].lower()

            # extract params
            L = re.search(r"L=([\deE\+\-\.]+)", line)
            W = re.search(r"W=([\deE\+\-\.]+)", line)
            nfin = re.search(r"nfin=(\d+)", line)
            X = re.search(r"\$X=([\deE\+\-\.]+)", line)
            Y = re.search(r"\$Y=([\deE\+\-\.]+)", line)

            L = float(L.group(1)) if L else None
            W = float(W.group(1)) if W else None
            nfin = int(nfin.group(1)) if nfin else 1
            X = float(X.group(1)) if X else None
            Y = float(Y.group(1)) if Y else None

            W_eff = W * nfin if W else None

            tr_type = "PMOS" if "pmos" in model else "NMOS"

            tr = {
                "name": name,
                "type": tr_type,
                "net_drn": drain,
                "net_src": source,
                "gate_net": gate,
                "net_bulk": bulk,
                "L": L,
                "W": W,
                "nfin": nfin,
                "W_norm": W_eff,
                "poly_x": X,
                "poly_y": Y
            }

            current_transistors.append(tr)

    return cells


# Example usage
cells = parse_spice(r"asap7sc7p5t_28\CDL\xAct3D_extracted\asap7sc7p5t_28_L.sp")

# Convert to token embedding schema
dataset = {}
for cname, trans in cells.items():
    row_P, row_N = [], []
    for tr in trans:
        entry = {
            "W_norm": tr["W_norm"],
            "poly_parity": int(tr["poly_x"] / 0.054) % 2 if tr["poly_x"] else 0,
            "net_src": tr["net_src"],
            "net_drn": tr["net_drn"],
            "gate_net": tr["gate_net"],
        }
        if tr["type"] == "PMOS":
            row_P.append(entry)
        else:
            row_N.append(entry)

    dataset[cname] = {
        "prefix": {
            "poly_pitch": None,
            "od_min_space": None,
            "cell_height_tracks": None,
            "poly_dir": 1
        },
        "row_P": row_P,
        "row_N": row_N
    }

# 存成 JSON 檔案
output_file = "preprocess/asap7_dataset.json"
with open(output_file, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"✅ 已存成 {output_file}")
