
# Transistor-Level Placement (Two-Row, Column-Aligned)

這個套件把助教提供的 chip-level 參考架構，改造成 **電晶體層級（單一 PMOS 列 + 單一 NMOS 列）** 的環境：
- **第一優先**：最小化 diffusion break
- **次要**：增加 shared diffusion length、減少 *非必要* dummy gate
- 欄位(column)對齊、pair（PMOS↔NMOS）鏡射同欄放置

## 檔案說明
- `parser_transistor.py`：讀取簡化 JSON（比 SPICE 容易驗證），輸出 features/adj/netlist… 供 GNN/Policy 使用。
- `env_transistor.py`：兩排欄位環境（Gym），動作=選一顆裝置；如果有 pair，環境會同欄放置（鏡射）。
- `policy_bias_utils.py`：從 env 生成 `(N,2)` 的 next-pos 幾何偏置，供 Transformer 注意力使用。
- `export_transistor_csv.py`：把結果輸出為 CSV（row/column/x/y/orient 等欄位）。
- `train_transistor_sb3.py`：用 SB3 的 PPO 訓練；若要用你的 TransformerPolicy/Value 可換掉 `policy` 與 `policy_kwargs`。

## 快速開始（僅示範流程，不強制執行）
```python
from train_transistor_sb3 import train
model = train("sample_circuit.json", total_timesteps=50000)
```

## 設計重點
- **Reward**：`-w_break * ΔBreaks + w_share * ΔShared - w_dummy_eff * ΔEffDummy + w_hpwl * Δ(-HPWL)`，以「上一個 episode 同步步數」為基準比較（和參考碼的 HPWL 機制對齊）。
- **Policy 偏置**：`next_pos = (col*poly_pitch, y_row)`，可進一步對「可共享的候選」降低距離，強化共享。
- **Value 特徵**：可把 break/shared/dummy/填充率等狀態當成標量拼到圖級嵌入。

## 與 chip-level 參考碼整合
- 保留 GNN/Transformer/Value 與 SB3 的訓練骨架（相容 API）。
- 只改動：輸入（JSON 取代 LEF/DEF）、環境（兩排欄位、pair 同欄）、reward 與匯出。
