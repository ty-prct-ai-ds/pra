import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def generate_shapes(shape, size=64):
    img = np.zeros((size, size), dtype=np.uint8)
    center = (size//2, size//2)
    if shape == 'circle':
        cv2.circle(img, center, size//3, 255, -1)
    elif shape == 'square':
        cv2.rectangle(img, (size//4, size//4), (3*size//4, 3*size//4), 255, -1)
    elif shape == 'triangle':
        pts = np.array(
            [
                [size//2, size//5], [size//5, size//2], [4*size//5, 4*size//5]
            ], np.int32
        )
        cv2.fillPoly(img, [pts], 255)
    return img


def create_dataset(num_samples=100):
    shapes = ["circle", "square", "triangle"]
    X, y = [], []

    for i, shape in enumerate(shapes):
        for _ in range(num_samples):
            img = generate_shapes(shape)
            X.append(img.flatten())
            y.append(i)
    return np.array(X), np.array(y)


X, y = create_dataset(300)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(X_test[i].reshape(64, 64), cmap="gray")
    axes[i].set_title(f"Pred: {y_pred[i]}")
    axes[i].axis("off")
plt.show()
