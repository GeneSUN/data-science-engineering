# Advanced Interview Questions: Neural Network Architecture



## Architecture Design & Understanding

1. How does increasing depth vs. width affect a neural network's performance? When would you prefer one over the other?
2. How do skip connections (residual connections) help in very deep networks like ResNet?
3. What is the benefit of using bottleneck layers (e.g., in ResNet or autoencoders)?
4. What is layer normalization and how is it different from batch normalization? When is one preferred over the other?

---

## Information Flow & Training Behavior

1. What is gradient flow, and how do you inspect or debug it in deep networks?
2. How does the choice of activation function (ReLU, GELU, Swish) affect gradient propagation?
3. What are exploding activations, and how are they different from exploding gradients?
4. How does depth affect model expressiveness and risk of overfitting or underfitting?
5. What are residual and dense connections, and how do they differ?

---

## Layer Types & Variants

1. Compare fully connected layers vs. convolutional layers — when and why do you use each?
2. What are depthwise separable convolutions (e.g., in MobileNet)? How do they reduce computation?
3. How do attention layers differ from traditional dense layers in terms of computation and structure?
4. What are dilated convolutions and when are they useful (e.g., in time series or segmentation)?

---

## Architectural Innovations

1. What is the role of positional encoding in Transformer architectures?
2. How does a Transformer encoder differ from a decoder in terms of architecture and use case?
3. Why are transformers replacing RNNs in NLP? What architectural advantage do they provide?
4. How do you implement and use custom residual blocks in practice (e.g., PyTorch)?
5. What are the trade-offs between CNNs, RNNs, and Transformers for sequence modeling?

---

## Model Design Trade-offs

1. How do you decide on the number of neurons in each layer of a neural network?
2. What is the vanishing gradient problem in LSTMs, and how do gates help solve it?
3. Why might a deeper network not perform better even with more data?
4. How does using dropout in early vs. late layers affect regularization?
