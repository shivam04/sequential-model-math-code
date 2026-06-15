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
    scores = H @ q  # (T,d) * (d, 1)            # (T,)
    weights = softmax(scores)   # (T,)
    context = weights @ H    # (1, T) * (T, d) =  # (d,)
    return context, weights

def atten_hidden(s_t, c_t, Wc):
    concat = np.concatenate([c_t, s_t])
    return tanh(Wc @ concat)

def output_layer(s_tilde, W_y):
    """
    y_t = softmax(Wy s~_t)
    """
    return softmax(W_y @ s_tilde)

# ---------- SETUP ----------
Wx = np.array([[1, 1],
               [1, 1]])

Wh = np.array([[1, 1],
               [1, 1]])

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
s_t = H[-1]   # initialize decoder with last encoder state

print("\n=== DECODER ===")
print("s_0:", s_t)

# dummy previous token input (just for demo)
y_tprev = np.array([1, 0])

Ws = [[1,1],
      [1,1]]

Wsh = [[1,1],
       [1, 1]]

Wc = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1]])

W_out = np.array([[0.2, 0.1],
                  [0.1, 0.3],
                  [0.4, 0.2]])

for t in range(2):

    # hidden decoder state
    s_t = rnn_step(y_tprev, s_t, Ws, Wsh)
    c_t, alpha_t = attention_luong(s_t, H)

    print(alpha_t)
    print(c_t)

    stilde_t = atten_hidden(s_t, c_t, Wc)
    print(stilde_t)

    y_t = output_layer(stilde_t, W_out)
    print("output probs:", y_t)
    y_tprev = y_t
