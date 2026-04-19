# ============================================================
# AI-ASSISTED GENOMIC SELECTION PIPELINE (FULL NOTEBOOK)
# ============================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (8,6)

# =========================
# 2. LOAD DATA
# =========================
geno = pd.read_csv("../data/raw/genotype_matrix_sample.csv")
pheno = pd.read_csv("../data/raw/phenotype_traits_sample.csv")

print("Genotype shape:", geno.shape)
print("Phenotype shape:", pheno.shape)

print(geno.head())
print(pheno.head())

# =========================
# 3. DATA VALIDATION
# =========================
print("\nMissing values (Genotype):", geno.isnull().sum().sum())
print("\nMissing values (Phenotype):")
print(pheno.isnull().sum())

print("\nDescriptive Statistics:")
print(pheno.describe())

# =========================
# 4. TRAIT DISTRIBUTION
# =========================
pheno.drop(columns=["Genotype"]).hist(bins=10)
plt.suptitle("Trait Distribution")
plt.show()

# =========================
# 5. CORRELATION ANALYSIS
# =========================
traits = pheno.drop(columns=["Genotype"])
corr = traits.corr()

plt.figure()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Trait Correlation Matrix")
plt.show()

# =========================
# 6. MERGE DATA
# =========================
data = pd.merge(pheno, geno, on="Genotype")
print("\nMerged data shape:", data.shape)

# =========================
# 7. SNP SUMMARY
# =========================
snp_data = geno.drop(columns=["Genotype"])

print("Mean SNP value:", snp_data.mean().mean())
print("Variance SNP:", snp_data.var().mean())

# =========================
# 8. PCA ANALYSIS
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(snp_data)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_

print("\nExplained variance (first 5 PCs):")
print(explained_var[:5])

# =========================
# 9. SCREE PLOT
# =========================
plt.plot(explained_var, marker='o')
plt.xlabel("Principal Components")
plt.ylabel("Variance Explained")
plt.title("Scree Plot")
plt.show()

# =========================
# 10. PCA PLOT
# =========================
plt.figure()
plt.scatter(X_pca[:,0], X_pca[:,1])

for i, txt in enumerate(geno["Genotype"]):
    plt.annotate(txt, (X_pca[i,0], X_pca[i,1]))

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Plot")
plt.show()

# =========================
# 11. SAVE PCA RESULTS
# =========================
pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
pca_df["Genotype"] = geno["Genotype"]

pca_df.to_csv("../data/processed/pca_scores.csv", index=False)

# PCA Loadings
loadings = pd.DataFrame(
    pca.components_.T[:, :2],
    columns=["PC1", "PC2"],
    index=snp_data.columns
)

loadings.to_csv("../data/processed/pca_loadings.csv")

# =========================
# 12. GENOMIC SELECTION MODEL
# =========================

# Select trait
target_trait = "Yield_per_Plant"

X = data.drop(columns=["Genotype", target_trait])
y = data[target_trait]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 13. RR-BLUP (Ridge)
# =========================
rr_model = Ridge(alpha=1.0)
rr_model.fit(X_train, y_train)

y_pred_rr = rr_model.predict(X_test)
rr_acc = r2_score(y_test, y_pred_rr)

print("\nRR-BLUP Accuracy (R²):", round(rr_acc, 3))

# =========================
# 14. RANDOM FOREST
# =========================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_acc = r2_score(y_test, y_pred_rf)

print("Random Forest Accuracy (R²):", round(rf_acc, 3))

# =========================
# 15. RESULT COMPARISON
# =========================
results = pd.DataFrame({
    "Model": ["RR-BLUP", "Random Forest"],
    "Accuracy (R²)": [rr_acc, rf_acc]
})

print("\nModel Comparison:")
print(results)

# =========================
# 16. SAVE RESULTS
# =========================
results.to_csv("../data/processed/model_results.csv", index=False)

print("\nPipeline completed successfully!")
