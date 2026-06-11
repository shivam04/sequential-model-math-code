import numpy as np

# ---------- ACTIVATIONS ----------
def tanh(x):
    return np.tanh(x)

# ---------- SIMPLE RNN CELL ----------
def rnn_step(x, h_prev, Wx, Wh):
    return tanh(Wx @ x + Wh @ h_prev)

# ---------- SETUP ----------
dx, dh = 2, 2

Wx = np.array([[0.5, 0.1],
               [0.3, 0.2]])

Wh = np.array([[0.4, 0.1],
               [0.2, 0.3]])

# inputs: I love cat
X = [
    np.array([1, 0]),
    np.array([0, 1]),
    np.array([1, 1])
]

# ---------- ENCODER ----------
h = np.zeros((2,))

print("h0:\n", h);

encoder_states = []

for x in X:
    h = rnn_step(x, h, Wx, Wh)
    encoder_states.append(h)

encoder_states = np.array(encoder_states)

print("Encoder states H:\n", encoder_states)

# ---------- DECODER (NO ATTENTION) ----------
h_dec = encoder_states[-1]

print("h_dec 0:\n", h_dec)

for t in range(3):
    h_dec = rnn_step(np.array([1,0]), h_dec, Wx, Wh)
    print("Decoder step", t, ":", h_dec)