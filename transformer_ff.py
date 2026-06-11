import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    """
    x: (T, d_model)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


# -----------------------------
# Scaled Dot-Product Attention
# -----------------------------
def attention(Q, K, V):
    """
    Q: (T, d_k)
    K: (T, d_k)
    V: (T, d_v)
    """

    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])

    weights = softmax(scores, axis=-1)

    output = weights @ V

    return output, weights


# -----------------------------
# Self Attention Layer
# -----------------------------
def self_attention(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    print("Shapes Q:\n", Q.shape)
    print("Shapes K:\n", K.shape)
    print("Shapes V:\n", V.shape)

    out, attn = attention(Q, K, V)
    return out, attn


# -----------------------------
# Feed Forward Network (FFN)
# -----------------------------
def feed_forward(X, W1, b1, W2, b2):
    """
    FFN:
        FFN(x) = W2 * relu(W1 x + b1) + b2
    """

    hidden = np.maximum(0, X @ W1 + b1)   # ReLU
    print("ff hidden:\n", hidden.shape)
    output = hidden @ W2 + b2

    print("ff output:\n", output.shape)

    return output

# -----------------------------
# FINAL VOCAB OUTPUT LAYER
# -----------------------------
def vocab_projection(X, W_out, b_out):
    """
    X: (T, d_model)
    W_out: (d_model, vocab_size)
    b_out: (vocab_size,)
    """

    logits = X @ W_out + b_out     # (T, vocab_size)
    probs = softmax(logits, axis=-1)

    return logits, probs

# -----------------------------
# Transformer Block
# -----------------------------
def transformer_block(X, params):
    """
    X: (T, d_model)
    """

    Wq, Wk, Wv = params["Wq"], params["Wk"], params["Wv"]
    print("Shapes Wq:\n", Wq.shape)
    print("Shapes Wk:\n", Wk.shape)
    print("Shapes Wv:\n", Wv.shape)
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]

    print("Shapes W1:\n", W1.shape)
    print("Shapes W2:\n", W2.shape)

    # -------------------------
    # 1. Self Attention
    # -------------------------
    attn_out, attn_weights = self_attention(X, Wq, Wk, Wv)

    print("Attention:\n", attn_out)
    print("Weights:\n", attn_weights)

    # Residual + Norm
    X = layer_norm(X + attn_out)

    # -------------------------
    # 2. Feed Forward Network
    # -------------------------
    ff_out = feed_forward(X, W1, b1, W2, b2)

    # Residual + Norm
    X = layer_norm(X + ff_out)

    return X, attn_weights


# -----------------------------
# Example Run
# -----------------------------
if __name__ == "__main__":

    vocab_size = 1000

    np.random.seed(0)

    # 3 tokens, embedding dim = 4
    X = np.array([
        [1.0, 0.5, 1.0, 0.0],
        [0.0, 1.0, 0.5, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ])

    print("X shape:\n",X.shape)

    T, d_model = X.shape

    d_k = d_v = d_model
    d_ff = 8  # feed-forward hidden size

    # -------------------------
    # Attention weights
    # -------------------------
    Wq = np.random.randn(d_model, d_k) * 0.1
    Wk = np.random.randn(d_model, d_k) * 0.1
    Wv = np.random.randn(d_model, d_v) * 0.1

    # -------------------------
    # Feed Forward weights
    # -------------------------
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros((1, d_ff))

    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros((1, d_model))

    params = {
        "Wq": Wq,
        "Wk": Wk,
        "Wv": Wv,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    # -------------------------
    # Forward pass
    # -------------------------
    output, attn = transformer_block(X, params)

    print("\nInput:\n", X)
    print("\nAttention Weights:\n", attn)
    print("\nTransformer Output:\n", output)

    W_out = np.random.randn(d_model, vocab_size) * 0.01
    b_out = np.zeros((vocab_size,))

    print("Shapes W_out:\n", W_out.shape)


    logits, probs = vocab_projection(X, W_out, b_out)

    print("\nLogits shape:", logits.shape)   # (T, 1000)
    print("\nProbabilities shape:", probs.shape)

    # show top-5 predictions for first token
    top5 = np.argsort(probs[0])[-5:][::-1]
    print("\nTop-5 token IDs for token 0:", top5)
    print("Top-5 probabilities:", probs[0][top5])