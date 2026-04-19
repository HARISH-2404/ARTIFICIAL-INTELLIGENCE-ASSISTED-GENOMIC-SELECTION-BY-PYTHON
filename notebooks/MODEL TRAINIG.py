import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# LOAD
data = pd.read_csv("../data/processed/genotype_scaled.csv")
pheno = pd.read_csv("../data/processed/phenotype_cleaned.csv")

# TARGET
y = pheno["Yield_per_Plant"]
X = data

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# RR-BLUP
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_rr = ridge.predict(X_test)

# RF
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# SAVE
pd.DataFrame({
    "RRBLUP": y_pred_rr,
    "RF": y_pred_rf
}).to_csv("../data/processed/predictions.csv", index=False)
