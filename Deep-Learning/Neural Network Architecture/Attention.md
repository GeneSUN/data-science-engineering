# Attention Mechanism

Attention was created to solve the limitations of LSTM in encoder-decoder structures.

## Decomposing Attention: A Series of Questions

<details>
<summary><b>Q1: What is the limitation of previous method? How to better understand a word?</b></summary>

Not in isolation. We understand a word through its interaction with other words — meaning is borrowed from context.

This relationship is **dynamic**: "bank" means something different next to "river" vs "money". The same word, different meaning, depending on context.

LSTM captures context too, but **statically** — influence decays with distance. Early words fade.

Attention makes every word's relationship to every other word a **directly computable quantity**, regardless of distance.

</details>

---

<details>
<summary><b>Q2: What is attention?</b></summary>

Attention is a mechanism that lets each word dynamically look up information from every other word in the sequence — weighted by relevance, not position. The following questions unpack how it works step by step: how words are represented as search queries, why the key and value vectors are kept separate, how similarity scores are computed, and what the final output actually is.

---

<details>
<summary><b>Q2.1: How do we represent each word?(q,k,v)</b></summary>

Raw embeddings are not enough. The embedding of "it" is the same in every sentence — averaged over the whole corpus, context-free, and nearly semantically empty.

Instead, project each word into three learned vectors:

| Vector | Role | Question it answers |
|--------|------|---------------------|
| **Q** (Query) | what this word is looking for | "what do I need from other words?" |
| **K** (Key) | what this word offers to be found | "what am I findable by?" |
| **V** (Value) | what this word actually contributes | "what content do I carry?" |

"it" projected into a Query becomes a precise search signal: *"I am a pronoun — find my antecedent."* Raw embedding cannot do this.

**Why separate K and V?**

If $K = V$, the same vector must simultaneously be good at *being found* and *delivering content* — competing objectives.

Separating them gives each a dedicated job:
- **K** is shaped purely for retrieval — optimized to match the right queries
- **V** is shaped purely for content — carries the richest representation forward

Analogy: a vector store does not search against the raw chunk text. It searches against a compressed index (K), then returns the full chunk content (V).

| Vector store | Attention |
|---|---|
| Search query | Q |
| Index of chunk | K |
| Actual chunk content | V |
| Retrieved result | weighted sum of V |

</details>

---

<details>
<summary><b>Q2.2: How do we compute the relationship between words?</b></summary>

For each word $i$, compute its similarity to every other word using dot product between its Query and their Keys:

$$\text{score}(q_i, k_j) = q_i \cdot k_j$$

Then scale and normalize into weights via softmax:

$$\text{weight}_j = \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)$$

Example — sentence: *"The animal didn't cross the street because it was tired"*

```
score("it" → "animal") = 0.92   ← high match
score("it" → "street") = 0.11
score("it" → "tired")  = 0.21

weight("animal") = 0.72
weight("street") = 0.08
weight("tired")  = 0.20
```

</details>

---

<details>
<summary><b>Q2.3: How to represent the word using its relationship with others?</b></summary>

The output for word $i$ is a weighted sum of all Value vectors:

$$\text{output}_i = \sum_j \text{weight}_j \cdot v_j$$

For "it":
```
output("it") = 0.72 × V("animal") + 0.08 × V("street") + 0.20 × V("tired")
```

This output is a new, context-enriched representation of "it" — no longer the vague standalone embedding, but a vector that has borrowed meaning from "animal", "tired", and everything else in the sentence, weighted by relevance.

</details>

---

<details>
<summary><b>Q2.4: How do you actually compute Q, K, V for each word? Where do the weight matrices come from?</b></summary>

Each word starts as a raw embedding — a 512-dimensional vector from a learned lookup table:

```
"it"     → [0.2, 0.8, 0.3, ..., 0.1]   ← 512 numbers
"animal" → [0.9, 0.1, 0.7, ..., 0.4]
"street" → [0.1, 0.6, 0.2, ..., 0.8]
```

Q, K, V are computed by multiplying the embedding by three learned weight matrices:

$$Q_i = e_i \cdot W_Q, \quad K_i = e_i \cdot W_K, \quad V_i = e_i \cdot W_V$$

Where $W_Q, W_K, W_V \in \mathbb{R}^{512 \times 64}$. **These matrices are learned during training — that is where all the intelligence lives.**

For multi-head attention, each head has its own set of weight matrices:

```
                      head_1              head_2          ...
                   ──────────────      ──────────────
embedding("it") →  × WQ_1 → Q_1       × WQ_2 → Q_2    ...
                   × WK_1 → K_1       × WK_2 → K_2    ...
                   × WV_1 → V_1       × WV_2 → V_2    ...
```

Each word produces $h$ different Q, K, V triplets — one per head — each living in a different 64-dimensional subspace.

</details>

</details>

---

<details>
<summary><b>Q3: How is attention applied in the Transformer architecture?</b></summary>

The Transformer is built entirely on attention — no LSTM, no convolution. It uses attention in three distinct ways inside an encoder-decoder structure: self-attention within the encoder (read), masked self-attention within the decoder (generate), and cross-attention between them (connect). The following three questions unpack each part.

---

<details>
<summary><b>Q3.1: Is one attention head enough? (multi-head attention)</b></summary>

No. Consider "it" in *"The animal didn't cross the street because it was too tired"*. "it" must simultaneously resolve:
- **Coreference** — which noun does "it" refer to? → "animal"
- **Causality** — why didn't it cross? → "tired"
- **Syntax** — what grammatical role does "it" play? → subject

A single attention head blends all of these into one weighted sum — the signals interfere and distinctions are lost.

**Solution: multi-head attention** — run $h$ attention operations in parallel, each with its own $W_Q^i, W_K^i, W_V^i$:

```
head 1 → specializes in coreference
head 2 → specializes in syntax
head 3 → specializes in causal relations
...
head 8 → specializes in something else
```

Nobody tells each head what to specialize in — it emerges from training.

**Dimensions:** with $h=8$ and $d_{model}=512$, each head operates in $512/8 = 64$ dimensions. Same total compute, 8 specialized perspectives.

**Final step:** concatenate all heads and project back:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_8) \cdot W_O$$

$W_O$ blends all 8 perspectives into one coherent 512-dim representation.

</details>

---

<details>
<summary><b>Q3.2: What is the general design of each encoder/decoder layer?</b></summary>

Each layer has two stacked components:

**Multi-head self-attention** (parallel across heads)
- Like an ensemble — each head has its own $W_Q^i, W_K^i, W_V^i$ and learns a different specialization
- All heads run simultaneously, then their outputs are concatenated
- Captures multiple relationship types at once (coreference, syntax, semantics...)

**Feed-forward network** (sequential, after attention)
- Applied independently to each position after attention
- Like boosting in spirit — each layer refines the previous layer's output
- Progressively builds deeper abstractions: surface patterns → syntax → semantics → reasoning

```
Input
  ↓
[Multi-Head Self-Attention]   ← parallel across h heads
  ↓
[Feed-Forward Network]        ← applied per position
  ↓
Output (fed into next layer)
```

Both encoder and decoder repeat this structure $N=6$ times.

</details>

---

<details>
<summary><b>Q3.3: What is the difference between encoder and decoder?</b></summary>

The difference lies in **what each attention layer is allowed to see**.

- **Encoder self-attention** — every word attends to every other word freely:
  ```
  "The animal didn't cross..."
   ↑    ↑      ↑      ↑
   all words can see all other words
  ```

- **Decoder self-attention** — each position can only attend to positions *before* it (masked):
  ```
  Generating y3: can see y1, y2 — cannot see y4, y5...
  ```

This is not arbitrary — it reflects the nature of the task:
- The encoder *reads* the full input, so it should see everything
- The decoder *generates* one token at a time, so it cannot look at future tokens it hasn't produced yet

Masking is implemented by setting future positions to $-\infty$ before softmax, so they receive zero attention weight.

</details>

---

<details>
<summary><b>Q3.4: How do encoder and decoder interact?</b></summary>

After the decoder's masked self-attention, there is a **cross-attention** layer where the two halves connect:

$$Q \leftarrow \text{decoder}, \quad K, V \leftarrow \text{encoder output}$$

```
Q → from decoder   ("what am I trying to generate next?")
K → from encoder   ("what input positions are available?")
V → from encoder   ("what content should I pull from?")
```

Example — translating French → English, generating "student":
- The decoder's current state forms a Query: *"I need the English word for the person being described"*
- It attends over all encoder hidden states (the full French sentence)
- Finds "étudiant" has the highest attention weight → borrows its content via V

This is the direct path between input and output that replaces the LSTM bottleneck — the decoder does not rely on a single compressed context vector; it re-reads the encoder at every step, focused on whatever is most relevant.

> The power of attention is not just Q, K, V as a mechanism. It is that Q, K, V enables direct, dynamic, lossless interaction between encoder and decoder — replacing the compressed bottleneck with a rich, flexible information highway.

</details>

</details>

---

## Comparison with RNN and CNN

<details>
<summary>Architecture Comparison: RNN vs CNN vs Self-Attention</summary>

| Property | RNN | CNN | Self-Attention |
|---|---|---|---|
| Parallelizable | ❌ | ✅ | ✅ |
| Long-range path length | O(n) | O(log n) | O(1) |
| Interpretable | ❌ | ❌ | ✅ |
| Notes | Sequential; slow for long dependencies | Needs many layers for distant positions | Complexity only bad when n >> d (rare) |

</details>

---

## Index

<details>
<summary><b>Positional Encoding</b></summary>

**Why positional encoding is needed**

Transformer's attention mechanism has a fundamental problem — it's order-blind:

```
"The cat sat on the mat"
"The mat sat on the cat"
```

Without positional encoding, the Transformer sees these as identical — same words, same embeddings, just shuffled. It has no idea which word came first. This is unlike RNN, which processes sequentially and inherently knows order.

**What positional encoding does**

It adds a position signal to each token's embedding:

```
Final representation = Word Embedding + Positional Encoding

Token "cat" at position 2:  [0.2, 0.8, 0.3] + [pos2 signal] = [0.5, 0.9, 0.1]
Token "cat" at position 7:  [0.2, 0.8, 0.3] + [pos7 signal] = [0.3, 0.6, 0.8]
                                    ↑                  ↑
                              same word          different position
                              same embedding     different final vector
```

Now the model can distinguish the same word at different positions.

**What the position signal actually looks like**

The original Transformer uses sine and cosine waves:

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Each position gets a unique, deterministic fingerprint vector:

```
Position 1:  [sin(1/1),   cos(1/1),   sin(1/100),   cos(1/100)  ...]
Position 2:  [sin(2/1),   cos(2/1),   sin(2/100),   cos(2/100)  ...]
Position 3:  [sin(3/1),   cos(3/1),   sin(3/100),   cos(3/100)  ...]
```

</details>

<details>
<summary><b>Encoder-Decoder</b></summary>

**Notation:**
- Input sequence: $x_1, x_2, \ldots, x_n$
- Encoder output: $z_1, z_2, \ldots, z_n$
- Decoder output: $y_1, y_2, \ldots, y_m$

---

**Why use encoder-decoder? Why not directly map $x \to y$?**

- **Reason 1:** Input and output may have different lengths ($n \neq m$)
- **Reason 2:** Output is autoregressive — each step depends on all previous outputs and the full input

The decoder factorizes the output probability as:

$$P(y_1, \ldots, y_m \mid x) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, x)$$

Meaning:

| Step | Predicts | Conditioned on |
|------|----------|----------------|
| $t=1$ | $y_1$ | $x$ |
| $t=2$ | $y_2$ | $x,\ y_1$ |
| $t=3$ | $y_3$ | $x,\ y_1, y_2$ |
| $\vdots$ | $\vdots$ | $\vdots$ |

- **Reason 3:** Separation of responsibilities — encoder focuses on understanding the input, decoder focuses on generating the output

</details>
