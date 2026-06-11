import numpy as np

# -----------------------------
# Softmax (numerically stable)
# -----------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)  # stability trick
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# -----------------------------
# Scaled Dot-Product Attention
# -----------------------------
def scaled_dot_product_attention(Q, K, V):
    """
    Q: (T_q, d_k)
    K: (T_k, d_k)
    V: (T_k, d_v)
    """

    # Step 1: compute raw attention scores
    # shape: (T_q, T_k)
    scores = Q @ K.T

    print("Scores:\n", scores)

    # Step 2: scale by sqrt(d_k)
    d_k = Q.shape[-1]
    scores = scores / np.sqrt(d_k)

    print("Scores scale:\n", scores)

    # Step 3: softmax to get attention weights
    # shape: (T_q, T_k)
    weights = softmax(scores, axis=-1)

    # Step 4: weighted sum of values
    # shape: (T_q, d_v)
    output = weights @ V

    return output, weights


# -----------------------------
# Self-Attention (Transformer block)
# -----------------------------
def self_attention(X, Wq, Wk, Wv):
    """
    X: (T, d_model)
    Wq, Wk, Wv: projection matrices
    """

    # Step 1: Linear projections
    # Q = XWq, K = XWk, V = XWv
    Q = X @ Wq   # (T, d_k)
    K = X @ Wk   # (T, d_k)
    V = X @ Wv   # (T, d_v)

    print("Shapes Q:\n", Q.shape)
    print("Shapes K:\n", K.shape)
    print("Shapes V:\n", V.shape)

    # Step 2: scaled dot-product attention
    output, weights = scaled_dot_product_attention(Q, K, V)

    return output, weights


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    # 3 tokens, embedding dim = 4
    X = np.array([
        [1.0, 0.0, 1.0, 0.0],  # token 1
        [0.0, 2.0, 0.0, 1.0],  # token 2
        [1.0, 1.0, 1.0, 1.0],  # token 3
    ])

    print("Shapes Input:\n", X.shape)

    d_model = 4
    d_k = 4
    d_v = 4

    # -----------------------------
    # Random learned projection matrices
    # (in real transformers: learned via backprop)
    # -----------------------------
    np.random.seed(42)

    Wq = np.random.randn(d_model, d_k) * 0.1
    Wk = np.random.randn(d_model, d_k) * 0.1
    Wv = np.random.randn(d_model, d_v) * 0.1

    print("Shapes Wq:\n", Wq.shape)
    print("Shapes Wk:\n", Wk.shape)
    print("Shapes Wv:\n", Wv.shape)

    # -----------------------------
    # Run self-attention
    # -----------------------------
    output, weights = self_attention(X, Wq, Wk, Wv)

    print("\nInput X:\n", X)
    print("\nAttention Weights (soft alignment matrix):\n", weights)
    print("\nOutput (contextual embeddings):\n", output)