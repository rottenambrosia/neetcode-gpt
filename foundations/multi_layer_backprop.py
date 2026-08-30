import numpy as np
from typing import List

class Solution:
    def forward_and_backward(self,
                             x: List[float],
                             W1: List[List[float]], b1: List[float],
                             W2: List[List[float]], b2: List[float],
                             y_true: List[float]) -> dict:
        # Convert lists to NumPy arrays
        x_arr = np.array(x, dtype=np.float64)
        W1_arr = np.array(W1, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        W2_arr = np.array(W2, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_true_arr = np.array(y_true, dtype=np.float64)

        # ----------------- FORWARD PASS -----------------
        # z = W @ x + b
        z1 = np.dot(W1_arr, x_arr) + b1_arr
        a1 = np.maximum(0, z1)
        z2 = np.dot(W2_arr, a1) + b2_arr
        
        # MSE Loss
        MSE = np.mean((z2 - y_true_arr) ** 2)

        # ----------------- BACKWARD PASS ----------------
        n = y_true_arr.shape[0]
        dz2 = (2.0 / n) * (z2 - y_true_arr)           # Shape: (out_dim,)
        
        # dW2 = dz2 (outer) a1 -> Shape: (out_dim, hidden_dim)
        dW2 = np.outer(dz2, a1)
        db2 = dz2
        
        # da1 = W2.T @ dz2 -> Shape: (hidden_dim,)
        da1 = np.dot(W2_arr.T, dz2)
        dz1 = da1 * (z1 > 0)                          # Shape: (hidden_dim,)
        
        # dW1 = dz1 (outer) x -> Shape: (hidden_dim, in_dim)
        dW1 = np.outer(dz1, x_arr)
        db1 = dz1

        # ----------------- FORMAT OUTPUT ----------------
        return {
            'loss': round(float(MSE), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }