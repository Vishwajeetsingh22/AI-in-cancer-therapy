# =========================================
# 1. Import required libraries
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# =========================================
# 2. Load dataset
# =========================================
df = pd.read_csv("lung_cancer.csv")

print(r"C:\Users\admin\Desktop\RESEARCH MANAGEMENT\rmpaper\lung_cancer.csv")
print("Shape:", df.shape)
print(df.head())

# =========================================
# 3. Drop ID column (if exists)
# =========================================
if "Patient_ID" in df.columns:
    df = df.drop("Patient_ID", axis=1)

# =========================================
# 4. Handle missing values
# =========================================
df = df.dropna()

# =========================================
# 5. Encode categorical columns
# =========================================
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# =========================================
# 6. Define features & target
# =========================================
TARGET = "Survival_1_Year"   # change if needed

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# =========================================
# 7. Train-test split (stratified)
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================
# 8. Train Random Forest
# =========================================
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

print("✅ Model Training Completed")

# =========================================
# 9. Predictions & Accuracy
# =========================================
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", accuracy)

print("\n📄 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================================
# 10. Confusion Matrix
# =========================================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================
# 11. Target Distribution
# =========================================
plt.figure()
sns.countplot(x=y)
plt.title("Target Class Distribution")
plt.show()

# =========================================
# 12. Correlation Heatmap
# =========================================
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================================
# 13. Feature Importance (Top 10)
# =========================================
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure()
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), X.columns[indices])
plt.title("Top 10 Important Features")
plt.xlabel("Importance Score")
plt.show()

# =========================================
# 14. ROC Curve & AUC Score
# =========================================
y_prob = rf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

print("\n📊 ROC-AUC Score:", auc_score)
