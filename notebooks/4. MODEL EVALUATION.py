import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

# LOAD
pred = pd.read_csv("../data/processed/predictions.csv")
pheno = pd.read_csv("../data/processed/phenotype_cleaned.csv")

y_true = pheno["Yield_per_Plant"].iloc[-len(pred):]

# RR
r2_rr = r2_score(y_true, pred["RRBLUP"])
rmse_rr = mean_squared_error(y_true, pred["RRBLUP"], squared=False)

# RF
r2_rf = r2_score(y_true, pred["RF"])
rmse_rf = mean_squared_error(y_true, pred["RF"], squared=False)

print("RRBLUP:", r2_rr, rmse_rr)
print("RF:", r2_rf, rmse_rf)
