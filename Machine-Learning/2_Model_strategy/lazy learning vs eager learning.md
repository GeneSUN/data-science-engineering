[Lazy Learning vs Eager Learning Algorithms in Machine Learning](https://www.analyticsvidhya.com/blog/2023/02/lazy-learning-vs-eager-learning-algorithms-in-machine-learning/)


Lazy: wait for query before generalizing
- k-Nearest Neighbor, Case based reasoning

Eager: generalize before seeing query
- Radial basis function networks, ID3,
  Backpropagation, NaiveBayes, …

Does it matter?
- Eager learner must create global approximation
- Lazy learner can create many local approximations
- if they use same H, lazy can represent more complex fns
  (e.g., consider H = linear functions)
