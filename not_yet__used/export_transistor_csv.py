
# export_transistor_csv.py
# Export env state to a CSV with transistor-specific fields.

import csv

def export_transistor_csv(path: str, env) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_name","device_type","row","column","x","y","orient","w","l","nf","shared_with","is_dummy"])
        for i, d in enumerate(env.devices):
            p = env.pos.get(i)
            if p is None:
                continue
            x, y, orient, row, col = p
            w.writerow([
                d["name"], d["type"], row, col, x, y, orient,
                d.get("w",1.0), d.get("l",0.05), d.get("nf",1),
                "", 0
            ])
