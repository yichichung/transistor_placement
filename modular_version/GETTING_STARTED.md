# 🚀 Getting Started with Modular Version

## Welcome!

This folder contains a complete modular restructure of the transistor placement system. Everything you need is right here!

---

## ⚡ Quick Start (3 Steps)

### Step 1: Navigate to this folder
```bash
cd modular_version
```

### Step 2: Run training
```bash
# Example: Train on circuits in parent directory
python train.py --env-dir ../circuits/ --timesteps 100000
```

### Step 3: Check results
```bash
# Results are saved to output/
ls output/best_by_cell/  # Best placements per cell
```

**That's it!** 🎉

---

## 📖 What's in This Folder?

```
modular_version/          👈 YOU ARE HERE
│
├── 📦 placement_env/     Environment & data processing
├── 🧠 placement_model/   Neural networks (GNN, Transformer)
├── 🔧 utils/             Callbacks & evaluation tools
│
├── 🚀 train.py          Start training here!
├── 📊 eval.py           Evaluate trained models
│
└── 📚 Documentation
    ├── GETTING_STARTED.md  (this file)
    ├── README.md           Quick reference
    ├── README_MODULES.md   Full guide
    └── MODULAR_STRUCTURE.md Technical details
```

---

## 🎯 Common Tasks

### 1️⃣ Train on a Single Circuit
```bash
python train.py --input-file path/to/circuit.json --timesteps 50000
```

### 2️⃣ Train on Multiple Circuits
```bash
python train.py --env-dir path/to/circuits_folder/ --timesteps 500000
```

### 3️⃣ Resume Training from Checkpoint
```bash
python train.py \
    --env-dir ../circuits/ \
    --resume-from output/multi_cell_model.pth \
    --timesteps 1000000
```

### 4️⃣ Evaluate a Trained Model
```bash
python eval.py \
    --model-path output/multi_cell_model.pth \
    --env-dir ../validation_circuits/ \
    --output-dir eval_results/
```

### 5️⃣ Monitor Training with TensorBoard
```bash
tensorboard --logdir output/tensorboard/
# Then open http://localhost:6006 in browser
```

### 6️⃣ Customize Reward Weights
```bash
python train.py \
    --env-dir ../circuits/ \
    --w-break 150 \
    --w-dummy 75 \
    --w-share 15 \
    --timesteps 500000
```

---

## 📊 Understanding Outputs

### After Training
```
output/
├── best_by_cell/
│   ├── cell1_best_placement.csv    ← Best result for cell1
│   ├── cell1_last_placement.csv    ← Latest result for cell1
│   ├── cell2_best_placement.csv
│   └── ...
│
├── tensorboard/                     ← TensorBoard logs
│   └── PPO_*/
│
└── multi_cell_model.pth            ← Trained model (save this!)
```

### CSV Format
Each placement file contains:
- **device_name**: Transistor name
- **device_type**: NMOS or PMOS
- **row**: Which row (PMOS/NMOS)
- **column**: Column index
- **x, y**: Physical coordinates
- **orient**: Orientation (R0/MX)
- **nf**: Number of fingers
- **pair_with**: Paired complementary device

---

## 🎓 Learning Path

### New to This Project?
1. Start with `README.md` for quick overview
2. Try training on a small circuit
3. Examine the output CSV files
4. Visualize results with TensorBoard

### Want to Understand the Code?
1. Read `MODULAR_STRUCTURE.md` for module breakdown
2. Check `placement_env/data_parser.py` for data loading
3. Check `placement_env/env.py` for RL environment
4. Check `placement_model/policy.py` for neural networks

### Want to Extend It?
1. Add new reward components in `placement_env/env.py`
2. Try different architectures in `placement_model/policy.py`
3. Add custom callbacks in `utils/callbacks.py`
4. All modules import cleanly - use as library!

---

## 🔧 Command Reference

### Training Arguments
```bash
python train.py \
    --env-dir <path>         # Directory with .json circuit files
    --timesteps 500000       # Total training steps
    --output-dir results/    # Where to save outputs
    --w-break 100            # Weight for diffusion breaks
    --w-dummy 50             # Weight for dummy devices
    --w-share 10             # Weight for shared diffusions
    --w-hpwl 2               # Weight for wire length
    --w-cdist 5              # Weight for pair distance
    --learning-rate 3e-4     # PPO learning rate
    --batch-size 64          # PPO batch size
    --resume-from model.pth  # Resume from checkpoint
```

### Evaluation Arguments
```bash
python eval.py \
    --model-path output/model.pth  # Trained model (required)
    --env-dir validation/          # Test circuits (required)
    --output-dir results/          # Where to save results
```

---

## 💡 Tips & Tricks

### 1. Start Small
```bash
# Try with just 10k timesteps first to make sure it works
python train.py --env-dir ../circuits/ --timesteps 10000
```

### 2. Use GPU if Available
```bash
# The code automatically uses CUDA if available
# You'll see "Using device: cuda" at start
```

### 3. Watch Training Live
```bash
# In another terminal:
tensorboard --logdir output/tensorboard/

# Metrics to watch:
# - placement/breaks (should decrease)
# - placement/hpwl (should decrease)
# - placement/shared (should increase)
```

### 4. Save Your Best Model
```bash
# The model is saved as output/multi_cell_model.pth
# Copy it somewhere safe!
cp output/multi_cell_model.pth ~/my_models/best_model_v1.pth
```

### 5. Evaluate on Held-Out Test Set
```bash
# Always evaluate on circuits not seen during training
python eval.py \
    --model-path output/multi_cell_model.pth \
    --env-dir ../test_circuits/ \
    --output-dir final_eval/
```

---

## ❓ FAQ

**Q: Where are the original files?**
A: The original `transistor_placement.py` is in the parent directory and still works!

**Q: Can I import these modules in my own code?**
A: Yes! Just add this directory to your Python path:
```python
import sys
sys.path.append('path/to/modular_version')
from placement_env import parse_transistor_json
from placement_model import GNNEncoder
```

**Q: How do I change the neural network architecture?**
A: Edit `placement_model/policy.py`. For example, change GNN layers, Transformer heads, etc.

**Q: Can I add new metrics?**
A: Yes! Edit `placement_env/env.py` and add to `_metrics()` function.

**Q: What if training is too slow?**
A: Try reducing `--n-steps` or `--batch-size`, or use a GPU.

**Q: How do I know if training is working?**
A: Check TensorBoard. The `placement/weighted_score` should generally decrease over time.

---

## 🎯 Next Steps

1. ✅ You're in the right folder (`modular_version/`)
2. ✅ Run your first training session
3. ✅ Check the outputs
4. ✅ Read the other docs for advanced features

---

## 📚 More Documentation

- **README.md**: Quick reference card
- **README_MODULES.md**: Complete feature guide
- **MODULAR_STRUCTURE.md**: Technical deep dive
- **FILE_LIST.txt**: What's in each file

---

## 🎉 You're All Set!

Everything is organized and ready to use. Just run:

```bash
python train.py --help
python eval.py --help
```

Happy training! 🚀
