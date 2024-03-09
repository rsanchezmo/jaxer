# Training 
## Output Discrete Grid
- Without **time2vec** --> OVERFIT || tested many configurations, but end up overfitting. Dropout even at 0.1. Try dropout of 0.5 and do not learn. 
- With **time2vec** --> UNDERFIT
- Try with different resolution (e.g. 1h and 4h) --> STILL OVERFITS
- TODO: REDUCE MODEL COMPLEXITY

## Output Mean/Distribution
- Unable to distinguish how good the model is --> achieve 2% of mape in train/test set. But it depends on the input data normalization. It is not the same 2% in a very noisy input with a very big std than 2% in a very flat and stable input (really low std). 
- TODO: check if adding std of the input makes better generalization results. 
- Must find a metric that measures how good the model is without these issues. That is why accuracy on ```output discrete grid``` is more explainable. 
- TODO: remove n trades (the variable is really correlated to the volume)