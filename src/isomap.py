import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from matplotlib import pyplot as plt


class MyIsomap:
    """
    A small class for implementing Isomap algorithm.
    """

    def __init__(self, n_components=2, n_neighbors=5):
        """
        Constructor.
        -----------
        Parameters:
        n_components: int, optional (default=2)
            Number of dimensions to reduce the data to.

        n_neighbors: int, optional (default=5)
            Number of neighbors to consider while computing pairwise distances.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.graph_ = None
        self.distance_graph_ = None
        self.embedding_ = None
        self.top_k_eigvecs_ = None
        self.top_k_eigvals_ = None

    def fit_transform(self, X):
        """
        Perform dimensionality reduction using Isomap algorithm

        Parameters:
        -----------
        X: array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        --------
        embedding_: array-like, shape (n_samples, n_components)
            The low-dimensional embedding of the input data points.
        """
        # Compute pairwise distances
        self.graph_ = kneighbors_graph(X, n_neighbors=self.n_neighbors).toarray()

        # Compute shortest path distances (faster version of F-W algorithm, which is reported in the next cell)
        D = sp.csgraph.shortest_path(self.graph_, method="FW")

        D = D**2

        # symmetrize!
        for i in range((D.shape[0])):
            for j in range((D.shape[0])):
                if D[i, j] != D[j, i]:
                    k = np.min([D[i, j], D[j, i]])
                    D[i, j] = k
                    D[j, i] = k

        self.distance_graph_ = D
        # Apply MDS to the distance matrix
        # double center
        n = D.shape[0]
        one_n = np.ones((n, n)) / n
        G = D - one_n.dot(D) - D.dot(one_n) + one_n.dot(D).dot(one_n)
        G = -(1 / 2) * G

        if np.any(np.isnan(G)) | np.any(np.isinf(G)):
            raise RuntimeError(
                "Graph is disconnected: please increase the number of neighbors (n_neighbors)"
            )

        eig_vals, eig_vecs = np.linalg.eig(G)
        eig_vals = np.where(eig_vals < 0, 0, eig_vals)
        sorted_indices = np.argsort(eig_vals)[::-1]
        sorted_eigvecs = eig_vecs[:, sorted_indices]
        sorted_eigvals = eig_vals[sorted_indices]

        # select the first k eigenvector(s) and project
        self.top_k_eigvecs_ = sorted_eigvecs[:, : self.n_components]
        self.top_k_eigvals_ = sorted_eigvals[: self.n_components]

        # Compute final embedding
        self.embedding_ = np.dot(
            self.top_k_eigvecs_, np.sqrt(np.diag(self.top_k_eigvals_))
        )

        return self.embedding_
