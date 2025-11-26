import os
import re
import json
import glob
from collections import defaultdict

# ================= CONFIG =================
INPUT_DIR = "./AutoGen"
OUTPUT_JSON = "expert_data.json"
# ==========================================


def parse_autocellgen_file_v4(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    solutions = defaultdict(list)
    current_sol = None

    sol_header = re.compile(r"-------- Solution (\d+) --------")

    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m_sol = sol_header.match(line)
        if m_sol:
            current_sol = int(m_sol.group(1))
            continue

        if current_sol is not None and ("NMOS :" in line or "PMOS :" in line):
            try:
                if "]" in line:
                    body = line.split("]", 1)[1]
                else:
                    body = line

                parts = body.split("PMOS :")
                nmos_part = parts[0].replace("NMOS :", "").strip()
                pmos_part = parts[1].strip() if len(parts) > 1 else ""

                def clean_name(s):
                    if not s:
                        return "dummy"
                    # 去掉 (nf) [nets]
                    s = s.split("(")[0].split("[")[0]
                    # 去掉雜質
                    s = s.strip(" ,")
                    # 過濾無效名稱
                    if not s or s.lower() == "dummy" or not (s.isalnum() or "_" in s):
                        return "dummy"
                    return s

                n_name = clean_name(nmos_part)
                p_name = clean_name(pmos_part)

                solutions[current_sol].append({
                    "nmos": n_name,
                    "pmos": p_name
                })
            except Exception:
                continue

    return solutions


def find_best_solution(solutions):
    best_sol_id = -1
    best_score = float('inf')
    best_seq = []
    best_metrics = {}

    for sol_id, cols in solutions.items():
        width = len(cols)
        dummy_count = 0

        # --- 步驟 1: 提取原始序列 ---
        raw_seq = []
        for col in cols:
            # 這裡我們只收集名字，先不管 dummy 統計
            n_name = col["nmos"]
            p_name = col["pmos"]

            # 統計 dummy (作為 Score 依據)
            if n_name == "dummy":
                dummy_count += 0.5
            if p_name == "dummy":
                dummy_count += 0.5

            # 加入原始序列 (NMOS 優先或 PMOS 優先不影響，RL Agent 會自己學)
            if n_name != "dummy":
                raw_seq.append(n_name)
            if p_name != "dummy":
                raw_seq.append(p_name)

        # --- 步驟 2: 智能去重 (Deduplication) ---
        # 我們希望保留順序，但移除連續重複
        # 例如: [MM1, MM1, MM2, MM2] -> [MM1, MM2]
        # 但保留: [MM1, MM2, MM1] -> [MM1, MM2, MM1] (如果是交錯的話)

        clean_seq = []
        seen_in_this_placement = set()  # 用來確保每個元件只出現一次 (針對 RL 環境限制)

        for name in raw_seq:
            # 如果這個名字跟上一個一樣，跳過 (連續重複)
            if clean_seq and clean_seq[-1] == name:
                continue

            # 如果這個名字已經出現過 (非連續重複)，
            # 這取決於您的 RL 環境：
            # 1. 如果您的環境支援「分段放置」(Split Action)，則保留。
            # 2. 如果您的環境是「一次放完」(One-shot)，則必須移除第二次出現。
            # 根據您的代碼，環境是一次放完 (placed[action]=1)，所以我們應該全局去重。
            if name in seen_in_this_placement:
                continue

            clean_seq.append(name)
            seen_in_this_placement.add(name)

        # --- 步驟 3: 評分 ---
        # 如果序列為空，視為無效
        if len(clean_seq) == 0:
            score = float('inf')
        else:
            score = 10 * width + dummy_count

        if score < best_score:
            best_score = score
            best_sol_id = sol_id
            best_seq = clean_seq
            best_metrics = {"width": width, "dummies": dummy_count}

    return best_seq, best_metrics


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Directory {INPUT_DIR} not found.")
        return

    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"Found {len(files)} files. Starting V4 Parsing (Deduplication)...")

    expert_database = {}

    for fpath in files:
        cell_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            solutions = parse_autocellgen_file_v4(fpath)
            if not solutions:
                continue

            best_seq, metrics = find_best_solution(solutions)

            if not best_seq:
                continue

            expert_database[cell_name] = {
                "sequence": best_seq,
                "metrics": metrics
            }

        except Exception as e:
            print(f"Error parsing {cell_name}: {e}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(expert_database, f, indent=2)

    print(f"Done. Expert data saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
