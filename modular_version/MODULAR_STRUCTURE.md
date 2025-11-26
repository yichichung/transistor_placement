# Modular Restructure Documentation

## 📁 Project Structure

```
transistor_placement/
├── placement_env/              # Environment & Data Processing
│   ├── __init__.py
│   ├── data_parser.py         # JSON parsing & graph construction
│   └── env.py                 # RL environment definitions
│
├── placement_model/            # Neural Network Architectures
│   ├── __init__.py
│   └── policy.py              # GNN, Transformer, Value Net, SB3 Policy
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── callbacks.py           # Training callbacks (TensorBoard, Best tracking)
│   └── utils.py               # Evaluation utilities
│
├── train.py                    # Training entry point
├── eval.py                     # Evaluation entry point
└── transistor_placement.py     # Original monolithic file (unchanged)
```

## 📦 Module Breakdown

### 1. `placement_env/` - Environment & Data Processing

**Focus:** Convert JSON files to graph structures, define RL state/reward/actions

#### `placement_env/data_parser.py`
| Class/Function | Purpose | Source Lines |
|----------------|---------|--------------|
| `TransistorNode` | Dataclass for transistor nodes | 45-56 |
| `POWER_NET_NAMES` | Power/ground net constants | 58 |
| `MOS_TYPES` | NMOS/PMOS type constants | 38 |
| `_is_power_net()` | Check if net is power/ground | 61-64 |
| `_want_pin_token()` | Filter S/D/G pins (exclude bulk) | 67-74 |
| `build_graph_from_nets()` | Build adjacency matrix from netlist | 77-116 |
| `parse_transistor_json()` | Main JSON parsing function | 119-234 |

#### `placement_env/env.py`
| Class/Function | Purpose | Source Lines |
|----------------|---------|--------------|
| `TransistorPlacementEnv` | Core RL environment | 446-785 |
| `RandomMultiCellEnv` | Multi-cell wrapper for training | 787-824 |
| `_normalize_pair_map()` | Validate NMOS-PMOS pairing | 486-510 |
| `_check_valid_placement()` | Private method (implicit in step) | - |
| `_compute_metrics()` | Calculate HPWL, breaks, shared, dummy | 620-634 |
| `_get_obs_and_info()` | Generate observation | 636-640 |

### 2. `placement_model/` - Neural Networks

**Focus:** Define GNN, Transformer, and PPO policy architectures

#### `placement_model/policy.py`
| Class/Function | Purpose | Source Lines |
|----------------|---------|--------------|
| `GNN_output_dim` | Global constant (128) | 37 |
| `GCNLayer` | Single GCN layer | 240-246 |
| `GNNEncoder` | 3-layer GNN encoder | 249-261 |
| `PositionEncoder1D` | Fourier position encoding | 267-299 |
| `CrossAttentionBlock` | Transformer attention block | 302-320 |
| `TransformerPolicy` | Transformer policy network | 322-422 |
| `ValueNetwork` | Value function for PPO | 428-440 |
| `TransistorPolicySB3` | SB3 policy integration | 831-1007 |
| `_compute_embeddings()` | Compute GNN embeddings | 850-854 |
| `forward()` | Main forward pass | 887-942 |
| `evaluate_actions()` | PPO action evaluation | 948-994 |

### 3. `utils/` - Utilities

**Focus:** Training callbacks, evaluation, and result export

#### `utils/callbacks.py`
| Class/Function | Purpose | Source Lines |
|----------------|---------|--------------|
| `TqdmCallback` | Progress bar callback | 1013-1034 |
| `BestPerCellCallback` | Track best placement per cell | 1036-1228 |
| `_on_step()` | Main callback logic | 1141-1228 |

#### `utils/utils.py`
| Function | Purpose | Source Lines |
|----------|---------|--------------|
| `eval_all_cells_greedy()` | Greedy evaluation on all cells | 1481-1521 |

### 4. Execution Scripts

#### `train.py` - Training Entry Point
```bash
# Single cell training
python train.py --input-file circuits/cell1.json --timesteps 100000

# Multi-cell training
python train.py --env-dir circuits/ --timesteps 500000

# Resume training
python train.py --env-dir circuits/ --resume-from output/model.pth
```

**Key Functions:**
- `make_sb3_model()` - Create PPO model for single cell
- `make_multicell_model()` - Create PPO model for multiple cells
- `train_transistor_placement()` - Main training loop
- `main()` - CLI argument parsing

#### `eval.py` - Evaluation Entry Point
```bash
# Evaluate on validation set
python eval.py --model-path output/multi_cell_model.pth \
               --env-dir val_circuits/ \
               --output-dir eval_results/
```

**Key Functions:**
- `main()` - Load model, run greedy inference, export results

## 🔄 Dependency Flow

```
eval.py
  └── utils.eval_all_cells_greedy()
       └── placement_model.TransistorPolicySB3
            └── placement_env.RandomMultiCellEnv
                 └── placement_env.parse_transistor_json()

train.py
  ├── placement_env.parse_transistor_json()
  ├── placement_env.TransistorPlacementEnv / RandomMultiCellEnv
  ├── placement_model.GNNEncoder
  ├── placement_model.TransformerPolicy
  ├── placement_model.ValueNetwork
  ├── placement_model.TransistorPolicySB3
  └── utils.TqdmCallback / BestPerCellCallback
```

## 📝 Import Examples

```python
# Training script
from placement_env import parse_transistor_json, TransistorPlacementEnv, RandomMultiCellEnv
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork, TransistorPolicySB3
from utils import TqdmCallback, BestPerCellCallback

# Evaluation script
from placement_env import RandomMultiCellEnv
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork, TransistorPolicySB3
from utils import eval_all_cells_greedy

# Custom development
from placement_env.data_parser import parse_transistor_json, MOS_TYPES
from placement_model.policy import GNN_output_dim
```

## ✅ Key Design Principles

1. **Separation of Concerns**
   - `placement_env/`: Data + Environment
   - `placement_model/`: Neural Networks
   - `utils/`: Support Functions
   - Scripts: Entry Points

2. **No Code Changes to Original**
   - `transistor_placement.py` remains untouched
   - All new code in modular structure

3. **Clear Dependency Chain**
   - `placement_env` has no dependencies on other modules
   - `placement_model` depends on `placement_env`
   - `utils` depends on both
   - Scripts depend on all

4. **Easy Testing & Maintenance**
   - Each module can be tested independently
   - Clear API boundaries via `__init__.py`
   - Docstrings on all major functions

## 🚀 Migration Guide

### For Training
**Before:**
```bash
python transistor_placement.py --env-dir circuits/ --timesteps 100000
```

**After:**
```bash
python train.py --env-dir circuits/ --timesteps 100000
```

### For Evaluation
**Before:**
```bash
python transistor_placement.py --eval-all --model-path model.pth --env-dir val/
```

**After:**
```bash
python eval.py --model-path model.pth --env-dir val/
```

## 📊 Code Metrics

| Module | Files | Lines | Classes | Functions |
|--------|-------|-------|---------|-----------|
| placement_env | 2 | ~700 | 2 | 6 |
| placement_model | 1 | ~450 | 7 | 3 |
| utils | 2 | ~200 | 2 | 1 |
| Scripts | 2 | ~450 | 0 | 5 |
| **Total** | **7** | **~1800** | **11** | **15** |

## 🎯 Next Steps

1. **Testing**: Add unit tests for each module
2. **Documentation**: Add more detailed API docs
3. **Extensions**: Easy to add new reward functions, network architectures, etc.
4. **Integration**: Can now import as library in other projects
