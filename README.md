\section\*{Transistor Placement --- Graph Data Format}

\subsection\*{From \texttt{nets} to Graph (\texttt{build*graph_from_nets})}
將輸入的連線清單 \texttt{nets} 轉為圖：
\begin{enumerate}
\item 將每個 net 中的所有裝置（忽略 pin 名）做 \textbf{clique 擴張}：同一 net 內的裝置兩兩相連。
\item 對角線補上 self-loop：$A*{ii} \leftarrow 1$。
\item \textbf{列正規化}得到 $A^{(norm)}$：
\[
A^{(norm)}_{ij} \;=\; \frac{A_{ij}}{\sum*{k} A*{ik} + \epsilon}
\]
其中 $\epsilon \approx 10^{-8}$，避免除以 0。\par
最終以 \texttt{torch.FloatTensor} 儲存為 \texttt{adj}（形狀 $N\times N$）。
\end{enumerate}

\subsection\*{Parsing 單一 JSON（\texttt{parse_transistor_json})}
解析單一 cell JSON，並建出 \texttt{graph_data} 結構供 GNN/Transformer/PPO 使用。

\subsection\*{\texttt{graph_data} 結構}
\begin{verbatim}
graph_data = {
"devices": List[dict],
"nodes": List[TransistorNode],
"features": torch.FloatTensor (N, 6),
"adj": torch.FloatTensor (N, N),
"netlist": List[List[int]],
"movable_indices": List[int],
"pair_map": Dict[str, str],
"name2idx": Dict[str, int],
"grid": Dict[str, float],
"num_cells": int,
"cell_name": str
}
\end{verbatim}

\subsection\*{欄位說明}
\paragraph{(1) \texttt{devices}}
直接來自輸入 JSON 的 MOS 清單（每顆元件一筆）：
\begin{lstlisting}[language=json]
[
{"name":"MN0","type":"NMOS","w":1.0,"l":0.05,"nf":1,"vt":"SVT"},
{"name":"MP0","type":"PMOS","w":1.0,"l":0.05,"nf":1,"vt":"SVT"}
]
\end{lstlisting}

\paragraph{(2) \texttt{nodes}（\texttt{TransistorNode}）}
將 \texttt{devices} 投影成便於操作的資料類別；未放置時 $x/y=None$：
\begin{verbatim}
TransistorNode(name='MN0', device_type='NMOS', width=1.0, length=0.05,
nf=1, vt='SVT', x=None, y=None, is_pin=False)
\end{verbatim}

\paragraph{(3) \texttt{name2idx}}
元件名稱 $\rightarrow$ 圖節點索引（例）：
\begin{verbatim}
{"MN0":0, "MN1":1, "MN2":2, "MP0":3, "MP1":4, "MP2":5}
\end{verbatim}

\paragraph{(4) \texttt{features}（形狀：$(N, 6)$）}
每顆元件一列特徵：
\[
\underbrace{[\; \mathrm{onehot}_{\text{NMOS}},\; \mathrm{onehot}_{\text{PMOS}},\; nf,\; w,\; l,\; \deg(i)\;]}_{\text{共 6 維}}
\]
\begin{itemize}
\item NMOS/PMOS one-hot：各 1 維（共 2 維）
\item $nf, w, l$：各 1 維
\item $\deg(i)$：節點 $i$ 的\textbf{連結度}，由 \texttt{adj} 的非正規化版本計算：
\[
\deg(i) \;=\; \sum_{j \neq i} \mathbf{1}\{A\_{ij}=1\}
\]
\end{itemize}
\textbf{例}：若 MN0 與 3 顆裝置相連，則
\[
\mathrm{feat}(\mathrm{MN0}) = [1,\,0,\,1.0,\,1.0,\,0.05,\,3].
\]

\paragraph{(5) \texttt{adj}（形狀：$N\times N$）}
由 \texttt{nets} 經 \textbf{clique 擴張}與\textbf{列正規化}得到的鄰接矩陣，供 GCN 使用。GCN 前向為
\[
H^{(l+1)} \;=\; \sigma\!\big( A^{(norm)}\, H^{(l)}\, W^{(l)} \big).
\]

\paragraph{(6) \texttt{netlist}}
每個 net 對應的\textbf{節點索引列表}（已去除 pin 名）。\par
\textbf{例}（對應 \texttt{name2idx}）：
\begin{verbatim}
["MN0.D","MP0.D","Y"] -> [0, 3]
["MN1.G","MP2.G","A"] -> [1, 5]
["MP0.S","MP0.B","MP1.S","MP2.S","VDD"] -> [3, 4, 5]
\end{verbatim}

\paragraph{(7) \texttt{pair_map}}
P/N \textbf{對偶映射}（利於同欄放置與成本計算），例：
\begin{verbatim}
{"MN0":"MP0","MP0":"MN0","MN1":"MP2","MP2":"MN1","MN2":"MP1","MP1":"MN2"}
\end{verbatim}

\paragraph{(8) \texttt{grid}}
將欄位轉為座標的尺度參數：
\begin{verbatim}
{"row*pitch":0.27, "poly_pitch":0.054, "y_pmos":1.0, "y_nmos":0.0}
\end{verbatim}
座標轉換：$x = \mathrm{poly\_pitch}\times \mathrm{column}$，$y\in\{y*{pmos},y\_{nmos}\}$。

\paragraph{(9) 其他}
\texttt{movable_indices = [0..N-1]}（預設全可放），\texttt{num_cells = N}，\texttt{cell_name} 為來源檔名（去副檔名）。

\subsection\*{建圖時的建議過濾（可選）}
\begin{itemize}
\item \textbf{忽略電源網}：遇到 \texttt{VDD}/\texttt{VSS}/\texttt{VDDX}/\texttt{VSSX} 等，直接略過該 net（或設定 \texttt{allow_power=False}）。
\item \textbf{忽略 bulk 腳}：pin 名為 \texttt{.B} 不計入建邊，避免圖過度稠密。
\end{itemize}

\subsection\*{最小可執行範例}
\begin{lstlisting}[language=Python]
from pathlib import Path
gd = parse_transistor_json(Path("AND2.json"), verbose=True)

print(gd["features"].shape) # -> torch.Size([N, 6])
print(gd["adj"].shape) # -> torch.Size([N, N])

enc = GNNEncoder(in_dim=gd["features"].size(1)).to(device)
H = enc(gd["features"].to(device), gd["adj"].to(device)) # (N, 128)
\end{lstlisting}

the terminal meseage for

cdl2json:
python cdl2json.py \
 --in asap7sc7p5t.sp \
 --out-dir preprocess

dedup:
python dedup_and_bin_cells.py --root preprocess --apply

reframe.py:
python reframe.py \
 --source preprocess/out_cells/\_clean/bins \
 --target preprocess/out_cells/\_clean/bins \
 --split 80/10/10 \
 --copy

then:
用 val/ 做 eval / 調參
最後用 test/ 做 final eval

1. 用 train/ 訓練: 跑 PPO 訓練, 學一個 model（GNN + Transformer + PPO）
   python transistor_placement.py \
    --env-dir preprocess/out_cells/\_clean/bins/0-5/train \
    --timesteps 50000 \
    --output-dir runs/0-5 \
    --learning-rate 3e-4 \
    --n-steps 2048 \
    --batch-size 64 \
    --ent-coef 0.02

2. 用 val/ 來「評估」這個模型
   現在已經有一個學好的 model：runs/0-5/multi_cell_model.pth。
   接著想看看它在驗證資料 val/ 上表現如何 → 用 eval 模式：
   用 runs/0-5/multi_cell_model.pth 的權重
   讀 0-5/val 裡的 json，逐顆 cell 做 greedy placement
   把結果存到：runs/0-5/eval_results/\*.csv

python transistor_placement.py \
 --env-dir preprocess/out_cells/\_clean/bins/0-5/train \
 --timesteps 50000 \
 --output-dir runs/0-5 \
 --learning-rate 3e-4 \
 --n-steps 2048 \
 --batch-size 64 \
 --ent-coef 0.02

python transistor_placement.py \
 --env-dir preprocess/out_cells/\_clean/bins/6-9/train \
 --timesteps 30000 \
 --output-dir runs/6-9 \
 --learning-rate 3e-4 \
 --n-steps 2048 \
 --batch-size 64 \
 --ent-coef 0.02 \
 --resume-from /Users/ninachung/Documents/GitHub/transistor_placement/runs/0-5/multi_cell_model.pth

python transistor_placement.py \
 --eval-all \
 --env-dir preprocess/out_cells/\_clean/bins/0-5/val \
 --model-path runs/0-5/multi_cell_model.pth \
 --output-dir runs/0-5
