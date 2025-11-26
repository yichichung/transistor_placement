# Transistor Placement - Modular Version

This folder contains the complete modular restructure of the transistor placement system.

## 📁 Structure

```
modular_version/
├── placement_env/              # Environment & Data Processing
│   ├── __init__.py
│   ├── data_parser.py         # JSON parsing & graph construction
│   └── env.py                 # RL environments
│
├── placement_model/            # Neural Network Architectures
│   ├── __init__.py
│   └── policy.py              # GNN, Transformer, Value Net, SB3 Policy
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── callbacks.py           # Training callbacks
│   └── utils.py               # Evaluation utilities
│
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
│
└── Documentation
    ├── README.md              # This file
    ├── MODULAR_STRUCTURE.md   # Detailed module breakdown
    └── README_MODULES.md      # User guide
```

## 🚀 Quick Start

### 1. Training

```bash
cd modular_version

# Single cell training
python train.py --input-file ../path/to/cell.json --timesteps 100000

# Multi-cell training
python train.py --env-dir ../path/to/circuits/ --timesteps 500000

# Resume from checkpoint
python train.py --env-dir ../circuits/ --resume-from output/model.pth
```

### 2. Evaluation

```bash
# Evaluate on validation set
python eval.py --model-path output/multi_cell_model.pth \
               --env-dir ../validation_circuits/ \
               --output-dir eval_results/
```

### 3. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir output/tensorboard/
```

## 📦 Module Overview

### `placement_env/` - Environment Module
- **data_parser.py**: Parse JSON netlists into graph structures
- **env.py**: RL environment (Gym interface)
  - `TransistorPlacementEnv`: Single-cell environment
  - `RandomMultiCellEnv`: Multi-cell training wrapper

### `placement_model/` - Neural Network Module
- **policy.py**: All neural network architectures
  - `GNNEncoder`: Graph Neural Network (3-layer GCN)
  - `TransformerPolicy`: Attention-based policy network
  - `ValueNetwork`: Value function for PPO
  - `TransistorPolicySB3`: Stable-Baselines3 integration

### `utils/` - Utilities Module
- **callbacks.py**: Training callbacks
  - `TqdmCallback`: Progress bar
  - `BestPerCellCallback`: Track best placements per cell
- **utils.py**: Evaluation utilities
  - `eval_all_cells_greedy()`: Greedy inference on validation set

## 📝 Command Line Arguments

### Training (`train.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | None | Single cell JSON file |
| `--env-dir` | None | Directory with multiple cells |
| `--timesteps` | 100000 | Total training steps |
| `--output-dir` | ./output | Output directory |
| `--w-break` | 100.0 | Weight for diffusion breaks |
| `--w-dummy` | 50.0 | Weight for dummy devices |
| `--w-share` | 10.0 | Weight for shared diffusions |
| `--w-hpwl` | 2.0 | Weight for wire length |
| `--w-cdist` | 5.0 | Weight for pair distance |
| `--learning-rate` | 3e-4 | PPO learning rate |
| `--n-steps` | 2048 | PPO rollout steps |
| `--batch-size` | 64 | PPO batch size |
| `--ent-coef` | 0.02 | Entropy coefficient |
| `--resume-from` | None | Checkpoint to resume from |

### Evaluation (`eval.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to trained .pth file |
| `--env-dir` | (required) | Validation circuits directory |
| `--output-dir` | ./eval_results | Output directory |

## 📊 Output Files

### Training Outputs

```
output/
├── best_by_cell/
│   ├── cell1_best_placement.csv    # Best placement for cell1
│   ├── cell1_last_placement.csv    # Latest placement for cell1
│   └── ...
├── tensorboard/
│   └── PPO_*/                      # TensorBoard logs
└── multi_cell_model.pth            # Trained model weights
```

### Evaluation Outputs

```
eval_results/
└── placements/
    ├── val_cell1_best_placement.csv
    ├── val_cell2_best_placement.csv
    └── ...
```

## 🔗 Import Examples

```python
# Import environment
from placement_env import parse_transistor_json, TransistorPlacementEnv

# Import model components
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork

# Import utilities
from utils import BestPerCellCallback, eval_all_cells_greedy

# Parse a circuit
graph_data = parse_transistor_json("circuit.json", verbose=True)

# Create environment
env = TransistorPlacementEnv(graph_data, reward_cfg={
    "w_break": 100.0,
    "w_dummy": 50.0,
    "w_share": 10.0,
    "w_hpwl": 2.0,
    "w_cdist": 5.0
})
```

## 📈 Metrics Tracked

- **breaks**: Diffusion breaks (lower is better)
- **dummy**: Dummy devices needed (lower is better)
- **shared**: Shared diffusions (higher is better)
- **hpwl**: Half-perimeter wire length (lower is better)
- **col_dist**: Column distance between pairs (lower is better)

**Weighted Score** (lower is better):
```
Score = 100×breaks + 50×dummy - 10×shared + 2×hpwl + 5×col_dist
```

## 🎯 Usage Examples

### Example 1: Basic Training
```bash
python train.py \
    --env-dir ../circuits/ \
    --timesteps 500000 \
    --output-dir results/
```

### Example 2: Custom Reward Weights
```bash
python train.py \
    --env-dir ../circuits/ \
    --w-break 150 \
    --w-dummy 75 \
    --w-share 15 \
    --w-hpwl 3 \
    --w-cdist 8 \
    --timesteps 500000
```

### Example 3: Resume Training
```bash
python train.py \
    --env-dir ../circuits/ \
    --resume-from results/multi_cell_model.pth \
    --timesteps 1000000
```

### Example 4: Evaluation
```bash
python eval.py \
    --model-path results/multi_cell_model.pth \
    --env-dir ../validation/ \
    --output-dir eval_results/
```

## 🔧 Dependencies

Required Python packages:
- `torch` - PyTorch for neural networks
- `stable-baselines3` - PPO implementation
- `gymnasium` (or `gym`) - RL environment interface
- `numpy` - Numerical computations
- `tqdm` - Progress bars

Install with:
```bash
pip install torch stable-baselines3 gymnasium numpy tqdm
```

## ✅ Key Features

1. **Modular Design**: Clean separation of concerns
   - Environment logic separate from model logic
   - Easy to swap components

2. **Preserved Functionality**: All original features work
   - Multi-cell training
   - Reward configuration
   - Resume from checkpoint
   - TensorBoard logging

3. **Easy to Extend**:
   - Add new reward functions in `placement_env/env.py`
   - Try different architectures in `placement_model/policy.py`
   - Add custom callbacks in `utils/callbacks.py`

4. **Well Documented**:
   - Comprehensive docstrings
   - Clear module boundaries
   - Usage examples

## 📚 Documentation

- **README.md** (this file): Quick start guide
- **MODULAR_STRUCTURE.md**: Detailed module breakdown with code mappings
- **README_MODULES.md**: Comprehensive user guide with all features

## 🆚 Comparison with Original

**Original monolithic file:**
```bash
python ../transistor_placement.py --env-dir circuits/ --timesteps 100000
```

**New modular version:**
```bash
python train.py --env-dir ../circuits/ --timesteps 100000
```

Both produce identical results, but the modular version is:
- ✅ Easier to maintain
- ✅ Easier to test
- ✅ Easier to extend
- ✅ Better organized
- ✅ Reusable as library

## 📄 License

Same as original project.

---

**Note:** The original `transistor_placement.py` file in the parent directory remains unchanged and continues to work independently.
