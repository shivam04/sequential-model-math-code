import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def attention(q, H):
    scores = H @ q
    weights = softmax(scores)
    context = weights @ H
    return context, weights

# ---------- INPUT ----------
X = np.array([
    [1, 0],
    [0, 1],
    [1, 1]
])

# ---------- ENCODER ----------
Wx = np.array([[0.5, 0.1],
               [0.3, 0.2]])

Wh = np.array([[0.4, 0.1],
               [0.2, 0.3]])

Wy = np.array([[1, 0.5],
               [0.2, 1],
               [0.8, 0.3]])

h = np.zeros(2)
print("h0:\n", h);
H = []

for x in X:
    h = np.tanh(Wx @ x + Wh @ h)
    H.append(h)

H = np.array(H)

print("H\n", H)

# ---------- TAGGING WITH SELF-ATTENTION ----------
for t in range(len(X)):
    q = H[t]

    context, attn = attention(q, H)

    final_rep = H[t] + context

    print(f'\nfinal: {final_rep}')

    logits = Wy @ final_rep
    probs = softmax(logits)

    print(f"Token {t}")
    print("attention:", attn)
    print("probs:", probs)