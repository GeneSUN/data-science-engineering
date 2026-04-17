# Interview Questions: RNN & LSTM

---

## Conceptual

1. What problem does an RNN solve that an MLP cannot?
2. What is the hidden state in an RNN, and what does it represent?
3. Why does a standard RNN struggle with long sequences? What is the root cause?
4. What is the difference between the hidden state and the cell state in an LSTM?
5. LSTM uses three gates. What does each one control, and why are all three necessary?
6. How do LSTM gates solve the vanishing gradient problem that standard RNNs suffer from?
7. What is a GRU, and how does it differ from an LSTM? When would you prefer one over the other?

---

## Scenario-based

8. You train an RNN on sequences of length 5 and it works well. You then apply it to sequences of length 200 and performance degrades significantly. What is the most likely cause?

9. An LSTM is trained for sentiment analysis on movie reviews. It performs well on short reviews but poorly on long ones. What architectural change would you consider first?

10. You have a time series forecasting task. What are the trade-offs between using an LSTM vs. a Transformer?

11. Why were Transformers able to replace RNNs for most NLP tasks? What specific limitation of RNNs do they address?

12. An RNN processes a sentence word by word. A Transformer processes all words simultaneously. What does the RNN gain and lose compared to the Transformer in this design?
