import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

print(df.head())

df['quality_label'] = df['quality'].apply(
    lambda q: 'good' if q >= 7 else 'bad')
df['quality_binary'] = df['quality_label'].apply(
    lambda x: 1 if x == 'good' else 0)


features = df.columns[:-3]
for col in features:
    sns.histplot(data=df, x=col, hue='quality_label',
                 kde=True, stat="density", common_norm=False)
    plt.title(f'Distribution of {col} by Quality')
    plt.show()

x = df[features]
y = df['quality_binary']
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
