
## 1️⃣ Feature encoding methods (model-agnostic)

These transform categories into numbers before any model sees them.


### One-hot

### Target / mean encoding

### Frequency / Count encoding

Replace category with how often it appears.

| region | count |
| ------ | ----- |
| NYC    | 120k  |
| TX     | 90k   |
| WY     | 400   |

Why it works

- High-frequency categories are often structurally important
- Works surprisingly well with trees & linear models

When it fails
- Rare but important categories get washed out

### Hashing Trick

Instead of mapping categories → unique IDs → one-hot
You hash them into fixed-size bins.

```python
region → hash(region) mod 10000
```


## 2️⃣ Models that natively handle categorical features

| Model        | How it handles categories                    |
| ------------ | -------------------------------------------- |
| **CatBoost** | Ordered target statistics + permutation      |
| **LightGBM** | Categorical split based on optimal partition |
| **XGBoost**  | One-hot-like splits on category              |

### CatBoost (best for categories)

Internally:
- Uses target encoding
- But avoids leakage via permutation
- Learns category → numeric mapping while training


### LightGBM

It finds the best grouping of categories at each split.

Instead of:
```
region == "NYC"?
```

It learns:
```
region ∈ {NYC, SF, LA, BOS} ?
```

### XGBoost

Weaker categorical handling; often still needs one-hot or target encoding.


## 3️⃣ Deep Learning: Embeddings

### A. Entity Embeddings from supervised training


This is where categorical features become latent vectors.

Train a NN for churn or CTR.
Extract embedding layer → use as features for other models.

```
region_id → Embedding(1000 → 16)
# Each region becomes a 16-D learned vector.
```

### B. Graph-based encoding

If categories are connected (cell towers, routers, users):


```
User — Router — Region
```

Run:
- Node2Vec
- GraphSAGE
- GNNs

You get embeddings that encode:

- Geography
- Behavior
- Network structure

### C. Hierarchical encoding

```
Country → State → City → Cell → Sector
```
Encode Each level, Plus interactions









