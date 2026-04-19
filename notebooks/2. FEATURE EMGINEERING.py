# =========================================
# IMPORT
# =========================================
import pandas as pd
from sklearn.preprocessing import StandardScaler

# =========================================
# LOAD
# =========================================
geno = pd.read_csv("../data/raw/genotype_matrix_sample.csv")
pheno = pd.read_csv("../data/raw/phenotype_traits_sample.csv")

# =========================================
# MERGE
# =========================================
data = pd.merge(pheno, geno, on="Genotype")

# =========================================
# SPLIT FEATURES
# =========================================
X = data.drop(columns=["Genotype","Yield_per_Plant"])
y = data["Yield_per_Plant"]

# =========================================
# SCALE
# =========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# SAVE
X_scaled_df.to_csv("../data/processed/genotype_scaled.csv", index=False)
y.to_csv("../data/processed/phenotype_cleaned.csv", index=False)
