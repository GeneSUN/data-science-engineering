## How to Handle an Imbalanced Dataset?

	1. Resampling Techniques(Undersampling/Oversampling)
	2. SMOTE (Synthetic Minority Over-sampling Technique)
	3. Class Weighting in Models
	4. Use Different Evaluation Metrics


### when we down-sample the positive class and observe how it affects Precision, Recall, and F1-score
 
1️⃣ Initial Scenario (Before Down-Sampling)

Given:
  - 100 positive samples (P = 100)
  - 10 negative samples (N = 10)
  - Imbalanced dataset: 100 positives, 10 negatives (total = 110)
  - Assume the model is biased towards predicting everything as positive (a common issue in imbalanced datasets).
    
Predictions (before down-sampling):
  - The model predicts all 110 samples as positive.
  - True Positives (TP) = 100 (Correct positive predictions).
  - False Positives (FP) = 10 (Incorrect positive predictions).
  - False Negatives (FN) = 0 (No missed positives).
  - True Negatives (TN) = 0 (No correct negatives).

Metrics Before Down-Sampling:
	

###  Cross-validation iterators with stratification based on class labels.¶
Some classification problems can exhibit a large imbalance in the distribution of the target classes: for instance there could be several times more negative samples than positive samples. In such cases it is recommended to use stratified sampling as implemented in ```StratifiedKFold``` and ```StratifiedShuffleSplit``` to ensure that relative class frequencies is approximately preserved in each train and validation fold.
From <https://scikit-learn.org/stable/modules/cross_validation.html> <img width="610" height="146" alt="image" src="https://github.com/user-attachments/assets/287e2670-7375-4acb-a29b-fbd7cadf2a9a" />

	
	
	
	
	
	

