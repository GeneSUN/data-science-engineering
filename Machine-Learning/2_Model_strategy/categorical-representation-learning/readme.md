
# Handling Categorical Features

This README is a compact guide to **turning categorical variables into model-ready signals**.

[Notebook companion](https://colab.research.google.com/drive/1cVj3QHHRGIXYXj1JFIc0I8r-zZ4u22qx)

## Table of Contents

- [Handling Categorical Features](#handling-categorical-features)
  - [0) Quick mental model](#0-quick-mental-model)
  - [1) Encoders](#1-encoders)
    - [1.1 Unsupervised encoders (do not use `y`)](#11-unsupervised-encoders-do-not-use-y)
    - [1.2 Supervised encoders (use `y`) – be careful about leakage](#12-supervised-encoders-use-y--be-careful-about-leakage)
    - [1.3 Hierarchical encoding with machine learning models](#13-hierarchical-encoding-with-machine-learning-models)
  - [2) Category-specific models (native categorical handling)](#2-category-specific-models-native-categorical-handling)
    - [2.1 CatBoost](#21-catboost)
    - [2.2 LightGBM (native categorical)](#22-lightgbm-native-categorical)
    - [2.3 XGBoost](#23-xgboost)
  - [3) Deep Learning: Embeddings and Transformers](#3-deep-learning-embeddings-and-transformers)
    - [3.1 Embedding + MLP baseline (CategoryEmbeddingModel / EmbeddingNet)](#31-embedding--mlp-baseline-categoryembeddingmodel--embeddingnet)
    - [3.2 TabTransformer](#32-tabtransformer)
    - [3.3 FT-Transformer (Feature Tokenizer Transformer)](#33-ft-transformer-feature-tokenizer-transformer)
    - [3.4 High-level frameworks](#34-high-level-frameworks)
      - [PyTorch Tabular](#pytorch-tabular)
      - [FastAI Tabular](#fastai-tabular)
    - [3.5 Self-supervised pretraining for categorical embeddings](#35-self-supervised-pretraining-for-categorical-embeddings)
  - [4) Graph-based encoding](#4-graph-based-encoding)
    - [4.1 Common graph constructions](#41-common-graph-constructions)
    - [4.2 How to turn graphs into features](#42-how-to-turn-graphs-into-features)
    - [4.3 When graph encoding helps most](#43-when-graph-encoding-helps-most)
  - [5) Practical checklist (things that usually bite)](#5-practical-checklist-things-that-usually-bite)
  - [References](#references)


---

## 0) Quick mental model

Categorical features are tricky because they can be:

- **Nominal (unordered)**: `city`, `device_model`, `plan_name`
- **Ordinal (ordered)**: `tier` in `{low < mid < high}`
- **High-cardinality IDs**: `region_id`, `user_id`, `merchant_id` (often thousands to millions)
- **Rare / unseen categories** at inference time
- **Leakage-prone** if you use supervised encoders (anything that needs `y`)

A good workflow is:

1) **Start simple** (OHE for low-card)  
2) Move to **compact encodings** for high-card (target / hashing / count)  
3) Prefer **native categorical tree models** (CatBoost / LightGBM) when possible  
4) Use **embeddings / Transformers** when you need interactions and scale  
5) Consider **graph encodings** when categories have relational structure

---
## 1) Encoders

Encoders convert categories into numeric representations. Choose based on **cardinality**, **model family**, and **leakage risk**.

### 1.1 Unsupervised encoders (do not use `y`)

- **One-hot encoding (OHE)**: best for low-cardinality; can explode columns for high-card.
- **Ordinal/label encoding**: only safe when the category truly has order; otherwise it injects a fake numeric ordering.
- **Count / frequency encoding**: replace each category with its count (or count / N). Compact and fast.
- **Hashing encoding**: hash categories into a fixed number of columns. Handles huge cardinality; collisions are the trade-off.
- **Rare-category bucketing**: group categories with count < k into `__RARE__`.

**Minimal demo (using `category_encoders` as in the notebook):**

```python
import category_encoders as ce

cat_cols = ["region", "device", "plan", "weekday"]

ohe = ce.OneHotEncoder(cols=cat_cols, handle_unknown="value", use_cat_names=True)
ord_enc = ce.OrdinalEncoder(cols=cat_cols, handle_unknown="value", handle_missing="value")

cnt_enc  = ce.CountEncoder(cols=cat_cols, normalize=False, handle_unknown=0, handle_missing=0)
freq_enc = ce.CountEncoder(cols=cat_cols, normalize=True,  handle_unknown=0, handle_missing=0)

hash_enc = ce.HashingEncoder(cols=cat_cols, n_components=16)  # fixed width
```

### 1.2 Supervised encoders (use `y`) - be careful about leakage

These can be very strong for high-cardinality features, but they **must be fit without leaking the target**.

- **Target / mean encoding**: replace category with (smoothed) mean target.
- **CatBoost-style target stats**: target encoding with noise / ordering ideas.
- **WOE / log-odds encoding** (common in credit scoring): log odds of event vs non-event for each category.

```python
# Supervised encoders need y during fit

tgt_enc = ce.TargetEncoder(cols=cat_cols, smoothing=10.0)
cb_enc  = ce.CatBoostEncoder(cols=cat_cols, a=1.0, sigma=0.05, random_state=42)
```

**Leakage rule:** for supervised encoders, do **out-of-fold (OOF) encoding** on train (KFold) and only then transform test.

### 1.3 Hierarchical encoding with machine learning models

Use this when categories have **natural hierarchy or interactions**, e.g. `region -> city -> store`, or `region x plan`.

Practical patterns:

- **Multi-level target stats (hierarchical smoothing):**
  - `enc(region)` smoothed toward global mean
  - `enc(region, plan)` smoothed toward `enc(region)`
  - Back off gracefully for rare groups

- **Stacking / meta-feature encoding:**
  - Train a lightweight model using categorical-only encodings (OHE/target/hashing)
  - Use its **OOF predicted probability** as a single dense feature

- **Tree leaf index encoding:**
  - Train a small tree model
  - Use the **leaf id** as a new categorical feature (then OHE / embedding it)

These approaches often work well when the true signal is in **interactions** (e.g., `region x plan`) rather than isolated categories.

---
## 2) Category-specific models (native categorical handling)

Sometimes the best "encoder" is: **let the model do it**.

| Model        | How it handles categories                    |
| ------------ | -------------------------------------------- |
| **CatBoost** | Ordered target statistics + permutation      |
| **LightGBM** | Categorical split based on optimal partition |
| **XGBoost**  | One-hot-like splits on category              |

> Note: Modern XGBoost also supports categorical features with `enable_categorical=True` and can use **partition-based** splits or **onehot** splits depending on configuration (see XGBoost docs linked below).

### 2.1 CatBoost

Why it is popular for categorical data:

- No manual OHE for most cases
- Strong defaults on high-cardinality features
- Uses *ordered* target statistics to reduce leakage/overfit

**Minimal pattern (from the notebook):**

```python
from catboost import CatBoostClassifier, Pool

cat_cols = ["region", "device", "plan", "weekday"]
cat_feature_indices = [X_train.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_feature_indices)

model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.08,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=100,
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)
```

### 2.2 LightGBM (native categorical)

LightGBM can handle categorical features natively when you pass them as categorical dtype (or integer-coded + flagged).

**Minimal pattern (from the notebook):**

```python
from lightgbm import LGBMClassifier

cat_cols = ["region", "device", "plan", "weekday"]

X_train_lgb = X_train.copy()
X_test_lgb  = X_test.copy()

for c in cat_cols:
    X_train_lgb[c] = X_train_lgb[c].astype("category")
    X_test_lgb[c]  = X_test_lgb[c].astype("category")

model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=128,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X_train_lgb, y_train, eval_set=[(X_test_lgb, y_test)], eval_metric="auc")
```

### 2.3 XGBoost

Common choices:

- If you are not using native categorical support: **OHE** (low-card) or **target/hashing** (high-card)
- If using native categorical support: set `enable_categorical=True` and let XGBoost choose a categorical split strategy

(Details vary by XGBoost version; use the docs linked below.)

---
- **Classic approach**: one-hot encode (or hashing) then train XGBoost.
- **Native categorical**: set `enable_categorical=True` (works with integer/`category` typed inputs).

```python
from xgboost import XGBClassifier

# Option A: OHE outside -> standard XGB
xgb_ohe = XGBClassifier(tree_method="hist", eval_metric="logloss")

# Option B: native categorical (requires categoricals marked as category/int + enable flag)
xgb_cat = XGBClassifier(
    tree_method="hist",
    enable_categorical=True,
    eval_metric="logloss",
)
```

---

## 3) Deep Learning: Embeddings and Transformers

When you have:

- very high-cardinality categories
- lots of interactions across fields
- large training data

...embeddings often work better than explicit encoders.

### 3.1 Embedding + MLP baseline ("CategoryEmbeddingModel" / EmbeddingNet)

Idea:

1. Each categorical field has its own embedding table: `id -> R^d`
2. Concatenate all embeddings with numeric features
3. Feed to an MLP

This is exactly what the notebook's `EmbeddingNet` demonstrates.

```python
import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, n_num, emb_dims, hidden_sizes=(256, 128), dropout=0.15):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(vocab, dim) for vocab, dim in emb_dims])
        emb_out = sum(dim for _, dim in emb_dims)

        layers = []
        in_dim = n_num + emb_out
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # binary logit
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x).squeeze(1)
```

**Embedding dimension heuristic:** `dim = min(50, round(1.6 * cardinality**0.56))` (used in the notebook).

### 3.2 TabTransformer

TabTransformer treats each categorical field as a token and uses **self-attention** to learn contextual, interaction-aware embeddings.

### 3.3 FT-Transformer (Feature Tokenizer Transformer)

FT-Transformer tokenizes **both** numerical and categorical features into embeddings and applies Transformer blocks. This often performs strongly as a general tabular deep learning baseline.

### 3.4 High-level frameworks

#### PyTorch Tabular

- Docs: `https://pytorch-tabular.readthedocs.io/`
- Supports: `CategoryEmbeddingModel`, `TabTransformer`, `FTTransformerConfig`, and more.

Minimal config pattern (as in the notebook):

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
# (FTTransformerConfig lives under pytorch_tabular.models as well)

data_config = DataConfig(
    target=["y"],
    continuous_cols=["x1", "x2", "x3"],
    categorical_cols=["region", "device", "plan", "weekday"],
)

model_config = TabTransformerConfig(task="classification")
trainer_config = TrainerConfig(batch_size=1024, max_epochs=20)

model = TabularModel(data_config=data_config, model_config=model_config, trainer_config=trainer_config)
```

#### FastAI Tabular

FastAI provides a very practical tabular pipeline with categorical embeddings + training utilities.

### 3.5 Self-supervised pretraining for categorical embeddings

If labels are limited (or you want reusable embeddings), you can pretrain embeddings with a **masked feature modeling** objective (BERT-style):

- randomly mask some categorical fields
- train a Transformer to predict the masked tokens
- reuse learned embeddings for downstream models (LightGBM/CatBoost) or fine-tune end-to-end

---

## 4) Graph-based encoding

Use graph encodings when categories are *relational* rather than independent columns.

### 4.1 Common graph constructions

- **Co-occurrence graph**: nodes are categories, edge weight = how often two categories appear together in the same row/session.
- **Bipartite graph**: connect `sample/entity` nodes (users, sessions, transactions) to category nodes.
- **Heterogeneous graph**: different node/edge types (user, item, city, device, plan, etc.).

### 4.2 How to turn graphs into features

- **Node2Vec / DeepWalk**: learn embeddings from random walks.
- **Graph Neural Nets (GraphSAGE, GAT, RGCN)**: learn embeddings with message passing.
- Then:
  - replace each category id with its **graph embedding**
  - aggregate multiple category embeddings (sum/mean/attention)
  - feed into a downstream model (GBDT or neural net)

### 4.3 When graph encoding helps most

- recommender systems (user-item graphs)
- merchant/category networks
- telecom/network topology (device-router-cell relationships)
- cases where "similar" categories should share statistical strength

---

## 5) Practical checklist (things that usually bite)

- Split train/test **before** fitting any encoder.
- For target-based encoders, do **out-of-fold encoding** on train to avoid leakage.
- Always decide how to handle:
  - **unknown** categories at inference (`__UNK__`)
  - **rare** categories (`__RARE__`)
- Prefer compact methods for high-cardinality:
  - CatBoost / LightGBM native categorical
  - target encoding (OOF)
  - hashing
  - embeddings
- Monitor drift: category vocab changes are common in production.

---

## References

- Notebook: `categorical_modeling.ipynb`
- `category_encoders`: https://contrib.scikit-learn.org/category_encoders/
- CatBoost docs: https://catboost.ai/
- LightGBM categorical features: https://lightgbm.readthedocs.io/en/latest/Features.html
- XGBoost categorical tutorial: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- PyTorch Tabular: https://pytorch-tabular.readthedocs.io/
