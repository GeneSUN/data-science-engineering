# RNN-Based Architectures

MLP and CNN process each input independently — they have no memory of previous inputs. For sequential data (text, time series, speech), order matters. RNNs are designed to handle this.

---

## RNN (Recurrent Neural Network)

At each timestep, an RNN takes the current input **and** the hidden state from the previous step:

```
x₁ → [RNN] → h₁ → [RNN] → h₂ → [RNN] → h₃ → output
       ↑              ↑              ↑
      x₁             x₂             x₃
```

The hidden state $h_t$ acts as memory — it carries information from previous timesteps forward:

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**Problem:** The hidden state is a fixed-size vector overwritten at every step. Long sequences cause the early context to fade — the network forgets what happened many steps ago. This is the **vanishing gradient problem applied to time** — gradients shrink as they backpropagate through many timesteps.

---

## LSTM (Long Short-Term Memory)

LSTM solves this by introducing a **cell state** $C_t$ — a separate memory lane that runs alongside the hidden state, with explicit gates controlling what to remember and forget:

```
        forget  input   output
           ↓      ↓       ↓
h_{t-1} →[  f  ][  i  ][  o  ]→ h_t
           ↓      ↓
C_{t-1} →[ × ]→[ + ]→ C_t
```

Three gates, each learned:

| Gate | What it controls |
|------|-----------------|
| **Forget gate** $f_t$ | How much of the old cell state to erase |
| **Input gate** $i_t$ | How much new information to write into cell state |
| **Output gate** $o_t$ | How much of the cell state to expose as hidden state |

The cell state $C_t$ can carry information across hundreds of timesteps unchanged — the forget gate simply keeps it at 1, letting it pass through untouched. This is the gradient highway for time, analogous to skip connections in deep networks.

---

## RNN vs LSTM

| | RNN | LSTM |
|--|-----|------|
| Memory | Hidden state only | Cell state + hidden state |
| Long-range dependencies | Struggles | Handles well |
| Complexity | Simple | More parameters (3 gates) |
| Vanishing gradient | Severe | Largely solved |

---

## GRU (briefly)

GRU is a simplified LSTM — merges the forget and input gates into a single **update gate**, and removes the separate cell state. Fewer parameters, similar performance in practice. Common when training data is limited.

---

## The bigger picture

Both RNN and LSTM process sequences **one step at a time** — they can't parallelize across timesteps. This is why Transformers eventually replaced them for most NLP tasks: attention processes all positions simultaneously, making training much faster on modern hardware.
