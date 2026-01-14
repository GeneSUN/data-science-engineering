# Self-Supervised Representation Learning for Tabular / Categorical Data

> It borrows from **BERT-style** masked modeling, but adapted to categorical columns instead of words.

Where masked categorical pretraining really comes from

## Literature review

### BERT (Devlin et al. 2018)

🚀 The idea is borrowed from NLP, BERT(Bidirectional Encoder Representations from Transformers))

### Paper 1: TabTransformer: Tabular Data Modeling Using Contextual Embeddings - Huang et al., NeurIPS 2020

- First major paper to treat categorical features as tokens
- Uses Transformer self-attention
- Learns contextual embeddings of categories
  
Key ideas
- Each categorical column → embedding
- Attention learns interactions between categories
- Embeddings outperform one-hot & target encoding


### Paper 2: SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training- Somepalli et al., NeurIPS 2021

What SAINT does
- Treats a row as a “sentence”
- Treats features as “tokens”
- Uses self-supervised pretraining
- Then fine-tunes on downstream tasks

Self-supervised objectives
- Masked feature modeling
- Contrastive learning between rows

### 📄 Paper 3: TabNet (Google): Attentive Interpretable Tabular Learning -Arik & Pfister, AAAI 2021

Different architecture, but same philosophy:
- Learn representations of categorical features
- Avoid one-hot explosion
- Use attention to select features

📌 Not masked modeling, but representation learning for tabular.



### 📄 Paper 4: FT-Transformer, Revisiting Deep Learning Models for Tabular Data - Gorishniy et al., NeurIPS 2021

Why this matters
- Shows Transformers + embeddings can beat GBDTs when done correctly
- Introduces FT-Transformer (Feature Tokenizer)

📌 Your tokenizer + embedding layer = FT-style design















