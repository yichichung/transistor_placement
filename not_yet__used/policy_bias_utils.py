
# policy_bias_utils.py
# Build per-candidate next-position bias for the Transformer policy.

import numpy as np

def build_next_pos_bias_from_env(env) -> np.ndarray:
    """
    Create an (N, 2) array of next (x, y) positions for each candidate device, using:
      - x = current column * poly_pitch
      - y = row y (y_pmos or y_nmos) determined by device type
    """
    N = env.N
    x = env.grid["poly_pitch"] * env.col_idx
    next_pos = np.zeros((N, 2), dtype=np.float32)
    for i, d in enumerate(env.devices):
        y = env.grid["y_pmos"] if d["type"] == "PMOS" else env.grid["y_nmos"]
        next_pos[i] = [x, y]
    return next_pos
