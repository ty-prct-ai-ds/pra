import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import fetch_kddcup99

kdd = fetch_kddcup99(percent10=True)
x_raw = pd.DataFrame(kdd.data)
y_raw = pd.Series(kdd.target)

categorical_cols = [1, 2, 3]
X = x_raw.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

y = y_raw.apply(lambda x: 0 if x == b'normal' else 1)

pca = PCA(n_components=20)
X_reduced = pca.fit_transform(X_scaled)

iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X_reduced)

y_pred = np.where(y_pred == -1, 0, 1)

print(f"Accuracy: {accuracy_score(y, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
            "Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix - Anomaly Detection on KDD99")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
