training rows have $(x_1, ..., x_n)$, but a new case arrives with only a subset $(x_1, ..., x_{n-p})$

0. Rebuilding a smaller model on only $(x_1, ..., x_{n-p})$, but it throws away $(x_{n-p+1}, ..., x_{n} )$

## Treat missing features as missing values


### Impute the missing features (Multiple imputation)

In Bayesian modeling, you can treat missing covariates as latent variables and infer them jointly with the prediction. 

- PyMC has examples showing “auto-impute missing covariates during sampling.”
- You can also calculate the distribution of Y, given the distribution of unknown x1,x2,x3











