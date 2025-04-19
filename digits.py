import numpy as np
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy().astype(np.float32) / 255.0
y = mnist.target.astype(int)

pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

num_clusters = 10
gmm = GaussianMixture(n_components=num_clusters, covariance_type="full", random_state=42)
gmm.fit(X_reduced)

clusters = gmm.predict(X_reduced)

def plot_cluster(cluster_number, num_samples=10):
    idxs = np.where(clusters == cluster_number)[0][:num_samples]
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i, idx in enumerate(idxs):
        axes[i].imshow(X[idx].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Cluster {cluster_number}')
    plt.show()

for i in range(num_clusters):
    plot_cluster(i)
