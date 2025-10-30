import re

def extract_cell_info(sp_file, target_cell):
    transistors = []
    inside = False
    subckt_name = None

    with open(sp_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*"):
                continue

            # subckt start
            if line.lower().startswith(".subckt"):
                parts = line.split()
                name = parts[1]
                # 過濾掉 parasitic modules
                if name.startswith("PM_"):
                    inside = False
                    continue
                else:
                    subckt_name = name
                    if subckt_name == target_cell:
                        inside = True
                    else:
                        inside = False
                continue

            # subckt end
            if line.lower().startswith(".ends"):
                if inside:
                    break
                else:
                    continue

            # transistor line
            if inside and line.startswith("M"):
                parts = line.split()
                name = parts[0]
                drain, gate, source, bulk = parts[1:5]
                model = parts[5].lower()
                tr_type = "PMOS" if "pmos" in model else "NMOS"

                transistors.append({
                    "name": name,
                    "type": tr_type,
                    "drain": drain,
                    "gate": gate,
                    "source": source,
                    "bulk": bulk,
                    "model": model
                })

    return transistors


# 使用範例
cell_name = "AO221x1_ASAP7_75t_L"
sp_file = r"asap7sc7p5t_28\CDL\xAct3D_extracted\asap7sc7p5t_28_L.sp"

trans = extract_cell_info(sp_file, cell_name)

print(f"Cell {cell_name}:")
print(f"  NMOS: {sum(1 for t in trans if t['type']=='NMOS')}")
print(f"  PMOS: {sum(1 for t in trans if t['type']=='PMOS')}")
print("  Transistors:")
for t in trans:
    print(f"   {t['name']} ({t['type']}): D={t['drain']}, G={t['gate']}, S={t['source']}, B={t['bulk']}")
