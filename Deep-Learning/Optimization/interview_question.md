# Interview Questions: Deep Learning Optimization

---

## Activation Functions

1. Why does a neural network need activation functions? What happens if you remove them entirely?

2. Under what scenario would you choose each of the following, and why?
   - ReLU
   - Sigmoid
   - Tanh
   - Softmax

3. A 10-layer network using Sigmoid activations trains very slowly — early layers barely update. What is the most likely cause, and how would you fix it?

4. ReLU can suffer from "dying ReLU" — neurons that permanently output zero. Under what conditions does this happen, and what are your options?

---

## Initialization

5. Why can't you initialize all weights to zero in a neural network?

6. A network using ReLU activations has exploding gradients in the very first backward pass, before any training has occurred. What is the most likely cause?

7. When would you use He initialization over Xavier, and why?

---

## Normalization

8. What is the difference between normalizing weights and normalizing activations? Which does BatchNorm do?

9. A model trains well with large batch sizes but performs poorly when batch size is reduced to 4. Batch normalization is in use. What is the likely cause, and how would you fix it?

10. Why is LayerNorm preferred over BatchNorm in Transformer architectures?

---

## Gradient Issues

11. A 20-layer network converges on the output layer quickly, but the first few layers show almost no weight updates across hundreds of epochs. What is happening, and what are three ways to address it?

12. During training, the loss suddenly spikes to NaN after several stable epochs. What is the most likely cause, and how do you diagnose it?

13. What is the difference between vanishing gradients and exploding gradients in terms of symptoms and solutions?

---

## Optimizers & Batch Strategy

14. What is the difference between backpropagation and gradient descent? Are they the same thing?

15. A colleague says "we trained with SGD." Does this tell you what batch strategy they used? Why or why not?

16. Adam is described as combining momentum and RMSProp. What does each component contribute, and why does that make Adam more robust than vanilla SGD?

17. During mini-batch training of a classification network, training accuracy oscillates significantly across epochs and never stabilizes. What is the most likely cause?
    - A. The batch size is too large
    - B. The model has too few parameters
    - C. The learning rate is too high
    - D. Dropout rate is too low

18. You switch from Adam to SGD and notice training becomes much more sensitive to the learning rate. Why?

---

## Learning Rate Scheduling

19. A Transformer model diverges in the first 100 steps of training despite a reasonable learning rate. What scheduling technique would you add, and why?

20. What is the difference between step decay and cosine annealing? When would cosine annealing be preferred?

21. Why can a model trained with a fixed learning rate fail to reach the same final performance as one with a decaying schedule, even given the same number of epochs?

---

## Regularization

22. Dropout is applied during training but disabled during inference. Why? What adjustment is made to compensate?

23. A model has low training loss but high validation loss that increases over time. What are three techniques you would apply, and at what stage of the pipeline does each operate?

24. What is the difference between L1 and L2 regularization in terms of effect on weights? When would you prefer L1?

25. Early stopping halts training based on validation loss. What is the risk of stopping too early, and how do you mitigate it?

---

## Loss Landscape & Generalization

26. A very large batch size speeds up training but the final model generalizes worse than one trained with small batches. Why might this happen?

27. Local minima are often cited as a problem in deep learning. Is this actually the main concern in practice? What is more problematic?

28. Two models reach the same training loss. Model A was trained with mini-batch SGD, Model B with full-batch gradient descent. Which is more likely to generalize better, and why?
