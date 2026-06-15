import numpy as np

# ---------- ACTIVATIONS ----------
def tanh(x):
    return np.tanh(x)

def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---------- RNN CELL ----------
def rnn_step(x, h_prev, Wx, Wh):
    return tanh(Wx @ x + Wh @ h_prev)

# ---------- LUONG ATTENTION (DOT PRODUCT) ----------
def attention_luong(q, H):
    """
    q: (d,)
    H: (T, d)
    """
    scores = H @ q              # (T,)
    weights = softmax(scores)   # (T,)
    context = weights @ H       # (d,)
    return context, weights

# ---------- SETUP ----------
Wx = np.array([[0.5, 0.1],
               [0.3, 0.2]])

Wh = np.array([[0.4, 0.1],
               [0.2, 0.3]])

# Input sequence (3 timesteps, 2-dim embeddings)
X = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

# ---------- ENCODER ----------
h = np.zeros(2)
H = []

print("=== ENCODER ===")
print("h0:", h)

for x in X:
    h = rnn_step(x, h, Wx, Wh)
    H.append(h)

H = np.array(H)

print("\nEncoder Hidden States H:\n", H)

# ---------- DECODER ----------
h_dec = H[-1]   # initialize decoder with last encoder state

print("\n=== DECODER ===")
print("h0_dec:", h_dec)

# dummy previous token input (just for demo)
x_t = np.array([1, 0])

for t in range(4):

    # 1. Attention over encoder states
    context, attn = attention_luong(h_dec, H)

    # 2. RNN step (decoder)
    h_prev = h_dec
    h_tilde = rnn_step(x_t, h_prev, Wx, Wh)

    # 3. Luong-style fusion (common simple version)
    h_dec = np.tanh(h_tilde + context)

    print(f"\nStep {t}")
    print("Attention weights:", attn)
    print("Context vector:", context)
    print("Hidden state:", h_dec)