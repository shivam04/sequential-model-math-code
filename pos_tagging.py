import numpy as np

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------- INPUT ----------
X = np.array([
    [1, 0],  # I
    [0, 1],  # love
    [1, 1]   # cats
])

# ---------- RNN ----------
Wx = np.array([[0.5, 0.1],
               [0.3, 0.2]])

Wh = np.array([[0.4, 0.1],
               [0.2, 0.3]])

Wy = np.array([[1, 0.5],
               [0.2, 1],
               [0.8, 0.3]])  # 3 tags

h = np.zeros(2)
print("h0:\n", h);

for t in range(len(X)):
    h = np.tanh(Wx @ X[t] + Wh @ h)

    logits = Wy @ h
    probs = softmax(logits)

    print(f"\nWord {t}")
    print("hidden:", h)
    print("probs:", probs)