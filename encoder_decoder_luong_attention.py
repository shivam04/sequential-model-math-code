import numpy as np

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def rnn_step(x, h_prev, Wx, Wh):
    return tanh(Wx @ x + Wh @ h_prev)

def attention_luong(q, H):
    scores = H @ q
    weights = softmax(scores)
    context = weights @ H
    return context, weights

# ---------- SETUP ----------
Wx = np.array([[0.5, 0.1],
               [0.3, 0.2]])

Wh = np.array([[0.4, 0.1],
               [0.2, 0.3]])

X = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

# ---------- ENCODER ----------
h = np.zeros(2)

print("h0:\n", h)

H = []

for x in X:
    h = rnn_step(x, h, Wx, Wh)
    H.append(h)

H = np.array(H)

print("Encoder H:\n", H)

# ---------- DECODER ----------
h_dec = H[-1]

print("h0_dec:\n", h_dec)

for t in range(4):
    q = h_dec

    context, attn = attention_luong(q, H)

    h_dec = rnn_step(np.array([1, 0]), h_dec, Wx, Wh)
    h_dec = h_dec + context

    print(f"\nStep {t}")
    print("Attention:", attn)
    print("Context:", context)
    print("Hidden:", h_dec)