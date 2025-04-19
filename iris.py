import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])


print(df.head())

print("Unique Species: ")
print(df['species'].unique())

sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.suptitle("Iris Dataset Pairplot", y=1.02)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("Classification Report: ")
print(classification_report(y_test, y_pred, target_names=iris.target_names))


feature_importance = pd.Series(
    clf.feature_importances_, index=iris.feature_names)
print("\nFeature Importance (Pattern Distinction):")
print(feature_importance.sort_values(ascending=False))

feature_importance.sort_values().plot(kind="barh", title="Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
