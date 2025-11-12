
import re, json, argparse, pathlib
from collections import defaultdict

NUM_RE = re.compile(r'([+-]?[0-9]*\.?[0-9]+)\s*([kKmMuUnNpPfFaA]?)')

UNIT_SCALE = {
    '': 1.0, 'k': 1e3, 'K': 1e3, 'm': 1e-3, 'M': 1e6,
    'u': 1e-6, 'U': 1e-6, 'n': 1e-9, 'N': 1e-9, 'p': 1e-12, 'P': 1e-12,
    'f': 1e-15, 'F': 1e-15, 'a': 1e-18, 'A': 1e-18,
}

def parse_value(token, default):
    if token is None:
        return default
    tok = str(token)
    m = NUM_RE.fullmatch(tok) or NUM_RE.match(tok)
    if not m:
        # allow values like 81.0n (without = sometimes)
        try:
            return float(tok)
        except Exception:
            return default
    val, unit = m.group(1), m.group(2)
    try:
        v = float(val) * UNIT_SCALE.get(unit, 1.0)
        return v
    except Exception:
        return default

def split_params(rest):
    # parse key=value tokens, tolerate commas
    params = {}
    for kv in re.findall(r'(\w+)\s*=\s*([^\s,]+)', rest):
        k, v = kv
        params[k.upper()] = v
    return params

def model_to_type(model):
    m = model.lower()
    if m.startswith('p') or 'pmos' in m:
        return 'PMOS'
    if m.startswith('n') or 'nmos' in m:
        return 'NMOS'
    return 'NMOS'  # safe default

def parse_cdl_text(text):
    # handle line continuations: join lines starting with '+'
    lines_raw = text.splitlines()
    lines = []
    buf = ''
    for ln in lines_raw:
        s = ln.strip()
        if not s or s.startswith('*'):
            # comment/blank terminates buffer
            if buf:
                lines.append(buf)
                buf = ''
            continue
        if s.startswith('+'):
            buf += ' ' + s[1:].strip()
        else:
            if buf:
                lines.append(buf)
            buf = s
    if buf:
        lines.append(buf)

    subckts = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.upper().startswith('.SUBCKT'):
            parts = ln.split()
            if len(parts) < 2:
                i += 1
                continue
            name = parts[1]
            pins = parts[2:]
            body = []
            i += 1
            while i < len(lines) and not lines[i].upper().startswith('.ENDS'):
                if lines[i] and not lines[i].startswith('*'):
                    body.append(lines[i])
                i += 1
            subckts.append((name, pins, body))
        i += 1
    return subckts

def build_json_from_subckt(name, pins, body, row_pitch=0.27, poly_pitch=0.054):
    devices = []
    nets_map = defaultdict(list)  # net -> ["M1.D", ...]
    supply_aliases_vdd = {'VDD','VDDQ','VDDC','VDD1','VCCA','VCC','VPWR','VDDIO','VPB'}
    supply_aliases_vss = {'VSS','VSSA','VSSD','VSS1','VSSQ','GND','VGND','VSSIO','VNB'}

    # MOS line pattern: Mname D G S B model params...
    mos_re = re.compile(r'^[Mm]([^\s]*)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)(.*)$')
    for ln in body:
        m = mos_re.match(ln)
        if not m:
            continue
        name_suffix, d,g,s,b,model,rest = m.groups()
        dev_name = ('M' + name_suffix) if name_suffix else 'M'
        params = split_params(rest)

        # Width & Length
        w = parse_value(params.get('W', params.get('WFIN', 1.0)), 1.0)
        l = parse_value(params.get('L', 0.05), 0.05)

        # Number of fins / fingers
        # Accept keys: NFIN (ASAP7), NF, M, NFINGERS
        nf_token = params.get('NFIN', params.get('NF', params.get('NFINGERS', params.get('M', 1))))
        try:
            nf = int(round(float(parse_value(nf_token, 1))))
        except Exception:
            nf = 1

        dev_type = model_to_type(model)

        devices.append({
            "name": dev_name,
            "type": dev_type,
            "w": round(w, 12),
            "l": round(l, 12),
            "nf": nf,
            "vt": "SVT"
        })

        # record pin nets
        nets_map[d].append(f"{dev_name}.D")
        nets_map[g].append(f"{dev_name}.G")
        nets_map[s].append(f"{dev_name}.S")
        nets_map[b].append(f"{dev_name}.B")

    # normalize supplies
    def norm_net(n):
        up = n.upper()
        if up in supply_aliases_vdd: return 'VDD'
        if up in supply_aliases_vss: return 'VSS'
        return n

    nets = []
    for net, pins_list in nets_map.items():
        net_norm = norm_net(net)
        nets.append(pins_list + [net_norm])

    # build heuristic pair_map: pair NMOS/PMOS sharing the same GATE net
    gate_by_dev = {}  # dev -> gate net
    for net, pins_list in nets_map.items():
        for tok in pins_list:
            if tok.endswith('.G'):
                gate_by_dev[tok.split('.')[0]] = norm_net(net)

    n_by_gate = defaultdict(list)
    p_by_gate = defaultdict(list)
    for d in devices:
        g = gate_by_dev.get(d["name"])
        if not g: 
            continue
        (n_by_gate if d["type"]=="NMOS" else p_by_gate)[g].append(d["name"])

    pair_map = {}
    for gnet in set(n_by_gate.keys()) | set(p_by_gate.keys()):
        Ns = sorted(n_by_gate.get(gnet, []))
        Ps = sorted(p_by_gate.get(gnet, []))
        for i in range(min(len(Ns), len(Ps))):
            ndev, pdev = Ns[i], Ps[i]
            pair_map[ndev] = pdev
            pair_map[pdev] = ndev

    data = {
        "devices": devices,
        "nets": nets,
        "constraints": {
            "pair_map": pair_map,
            "row_pitch": row_pitch,
            "poly_pitch": poly_pitch,
            "y_pmos": 1.0,
            "y_nmos": 0.0
        }
    }
    return data

def main():
    ap = argparse.ArgumentParser(description="Convert CDL (.cdl) subckts to JSON for transistor_placement.py")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input .cdl files")
    ap.add_argument("--out-dir", required=True, help="Output folder for per-cell JSON")
    ap.add_argument("--row-pitch", type=float, default=0.27)
    ap.add_argument("--poly-pitch", type=float, default=0.054)
    ap.add_argument("--filter", help="Only export subckt names matching this regex (e.g., '^(INV|NAND|AND)')")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cdl in args.inputs:
        text = pathlib.Path(cdl).read_text(errors="ignore")
        subckts = parse_cdl_text(text)
        for name, pins, body in subckts:
            if args.filter and not re.search(args.filter, name):
                continue
            data = build_json_from_subckt(name, pins, body, row_pitch=args.row_pitch, poly_pitch=args.poly_pitch)
            out_path = out_dir / f"{name}.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[OK] {name} -> {out_path}  (devices={len(data['devices'])}, nets={len(data['nets'])}, nfin_used={[d['nf'] for d in data['devices'][:3]]}...)")

if __name__ == "__main__":
    main()
