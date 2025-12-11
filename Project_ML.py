import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ==============================================================
#  تحميل البيانات
# ==============================================================
df = pd.read_csv("creditcard.csv")

# Features فقط (من غير Class)
X = df.drop(columns=['Class'])
y = df['Class']


# ==============================================================
#  Standardization
# ==============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================================================
#  PCA كامل لحساب الـ Variance
# ==============================================================
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# طباعة أول 10 Components
for i in range(1, 10 + 1):
    print(f"{i} components explain {cumulative_variance[i-1] * 100:.2f}% variance")


# رسم الـ Cumulative Variance
plt.figure()
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()


# ==============================================================
#  Visualization باستخدام 2 Components
# ==============================================================
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

plt.figure()
plt.scatter(X_pca_2[:, 0], X_pca_2[:, 1], s=1)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("PCA Visualization (2D)")
plt.show()


# ==============================================================
#  Classification Model باستخدام 10 Components
# ==============================================================
pca_10 = PCA(n_components=10)
X_pca_10 = pca_10.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca_10, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n---- Model Performance ----")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
