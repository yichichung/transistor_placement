# Transistor Placement - Modular Version

This is a modular refactoring of the transistor placement system. The original monolithic file `transistor_placement.py` has been split into organized modules for better maintainability and reusability.

## Quick Start

### Training
```bash
# Single cell
python train.py --input-file circuits/cell1.json --timesteps 100000

# Multiple cells
python train.py --env-dir circuits/ --timesteps 500000 --output-dir output/

# Resume from checkpoint
python train.py --env-dir circuits/ --resume-from output/multi_cell_model.pth
```

### Evaluation
```bash
python eval.py --model-path output/multi_cell_model.pth \
               --env-dir validation_circuits/ \
               --output-dir eval_results/
```

## Directory Structure

```
.
├── placement_env/          # Environment & data processing
│   ├── __init__.py
│   ├── data_parser.py     # JSON → Graph conversion
│   └── env.py             # RL environment (Gym)
│
├── placement_model/        # Neural networks
│   ├── __init__.py
│   └── policy.py          # GNN, Transformer, PPO policy
│
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── callbacks.py       # Training callbacks
│   └── utils.py           # Evaluation functions
│
├── train.py               # Training entry point
├── eval.py                # Evaluation entry point
│
└── transistor_placement.py  # Original (unchanged)
```

## Module Details

### 1. `placement_env` - Environment Module

**Purpose:** Parse circuit netlists and define RL environment

**Key Components:**
- `TransistorNode`: Device data structure
- `parse_transistor_json()`: Convert JSON to graph
- `TransistorPlacementEnv`: Single-cell RL environment
- `RandomMultiCellEnv`: Multi-cell training wrapper

**Example Usage:**
```python
from placement_env import parse_transistor_json, TransistorPlacementEnv

# Parse circuit
graph_data = parse_transistor_json("circuit.json", verbose=True)

# Create environment
env = TransistorPlacementEnv(graph_data, reward_cfg, device)
obs, info = env.reset()
```

### 2. `placement_model` - Neural Network Module

**Purpose:** Define GNN encoder, Transformer policy, and PPO integration

**Key Components:**
- `GNNEncoder`: 3-layer Graph Convolutional Network
- `TransformerPolicy`: Attention-based policy
- `ValueNetwork`: State value estimator
- `TransistorPolicySB3`: Stable-Baselines3 integration

**Example Usage:**
```python
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork

encoder = GNNEncoder(in_dim=6, out_dim=128)
policy = TransformerPolicy(embed_dim=128)
value_net = ValueNetwork(embed_dim=128, extra_dim=5)
```

### 3. `utils` - Utilities Module

**Purpose:** Training callbacks and evaluation utilities

**Key Components:**
- `TqdmCallback`: Progress bar
- `BestPerCellCallback`: Track best placements per cell
- `eval_all_cells_greedy()`: Greedy evaluation

**Example Usage:**
```python
from utils import BestPerCellCallback, eval_all_cells_greedy

# During training
callback = BestPerCellCallback(env, output_dir, reward_cfg)
model.learn(total_timesteps=100000, callback=callback)

# Evaluation
eval_all_cells_greedy(env, model, output_dir)
```

## Training Arguments

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

## Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | (required) | Path to trained .pth file |
| `--env-dir` | (required) | Validation circuits directory |
| `--output-dir` | ./eval_results | Output directory |

## Output Files

### Training
```
output/
├── best_by_cell/                    # Best placements per cell
│   ├── cell1_best_placement.csv
│   ├── cell1_last_placement.csv
│   └── ...
├── tensorboard/                     # TensorBoard logs
│   └── PPO_*/
└── multi_cell_model.pth            # Trained model
```

### Evaluation
```
eval_results/
└── placements/
    ├── val_cell1_best_placement.csv
    ├── val_cell2_best_placement.csv
    └── ...
```

## CSV Output Format

Each placement CSV contains:

| Column | Description |
|--------|-------------|
| device_name | Transistor name |
| device_type | NMOS or PMOS |
| row | PMOS or NMOS row |
| column | Column index |
| x | X coordinate |
| y | Y coordinate |
| orient | Orientation (R0/MX) |
| w | Width |
| l | Length |
| nf | Number of fingers |
| pair_with | Paired device name |

## Metrics Tracked

- **Breaks**: Diffusion breaks (lower is better)
- **Dummy**: Dummy devices needed (lower is better)
- **Shared**: Shared diffusions (higher is better)
- **HPWL**: Half-perimeter wire length (lower is better)
- **ColDist**: Column distance between pairs (lower is better)

**Weighted Score** (lower is better):
```
Score = 100*breaks + 50*dummy - 10*shared + 2*hpwl + 5*col_dist
```

## Advanced Usage

### Custom Reward Weights
```bash
python train.py --env-dir circuits/ \
    --w-break 150 \
    --w-dummy 75 \
    --w-share 15 \
    --w-hpwl 3 \
    --w-cdist 8
```

### GPU Training
```bash
# Automatically uses CUDA if available
python train.py --env-dir circuits/ --timesteps 1000000
```

### TensorBoard Monitoring
```bash
tensorboard --logdir output/tensorboard/
```

## Dependency Diagram

```
┌─────────────────────────────────────────────┐
│              train.py / eval.py             │
└──────┬──────────────────┬──────────────┬────┘
       │                  │              │
       ▼                  ▼              ▼
┌──────────────┐   ┌──────────────┐   ┌──────┐
│placement_env │   │placement_model│   │utils │
└──────────────┘   └──────────────┘   └──────┘
       │                  │              │
       └──────────────────┴──────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │External Dependencies  │
         │ - torch               │
         │ - stable-baselines3   │
         │ - gymnasium           │
         │ - numpy               │
         └───────────────────────┘
```

## Migration from Original

The original `transistor_placement.py` is **unchanged** and still works:
```bash
python transistor_placement.py --env-dir circuits/ --timesteps 100000
```

New modular version:
```bash
python train.py --env-dir circuits/ --timesteps 100000
```

All functionality is preserved with better organization!

## Benefits of Modular Structure

✅ **Separation of Concerns**: Environment, model, and utilities are independent
✅ **Easier Testing**: Test each module separately
✅ **Better Reusability**: Import modules in other projects
✅ **Maintainability**: Easier to find and fix bugs
✅ **Extensibility**: Add new features without touching existing code
✅ **Documentation**: Each module is self-contained with docstrings

## License

Same as original project.
