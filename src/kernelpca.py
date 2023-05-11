class MyRBFKernelPCA:
    """
    A small class to perform gaussian kernl PCA
    ------------
    Attributes:
    n_components: int, default is None
        number of principal components we want to project on
    gamma: float
        kernel coefficient (gaussian)
    top_k_eigvals_: None
        projection matrix
    """

    def __init__(self, n_components=None, gamma=1.0):
        """
        Constructor for MyRBFKernelPCA.
        ----------
        Parameters:
        n_components : int (optional)
            The number of principal components we want to project on (default is None, meaning to project on all components).
        gamma: float
            kernel coefficient (gaussian)
        """
        self.n_components = n_components
        self.gamma = gamma

    def fit_transform(self, X):
        """
        Fit the kernel PCA model.
        ----------
        Parameters:
        X : numpy array
            The data to fit the PCA model and transform, required in the form: (n x p) where n = number of observations and p = number of features
        ----------
        Returns:
        projected_data : numpy array
            The projected data.
        """
        # compute the squared Euclidean distances
        D = (
            np.sum(X**2, axis=1).reshape(-1, 1)
            + np.sum(X**2, axis=1)
            - 2 * np.dot(X, X.T)
        )

        # compute the RBF kernel matrix
        K = np.exp(-self.gamma * D)

        # double center the kernel matrix
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        eig_vals, eig_vecs = np.linalg.eig(K_centered)
        eig_vals = np.where(eig_vals < 0, 0, eig_vals)
        sorted_indices = np.argsort(eig_vals)[::-1]
        sorted_eigvecs = eig_vecs[:, sorted_indices]
        sorted_eigvals = eig_vals[sorted_indices]

        # select the first k eigenvector(s) and project
        top_k_eigvecs = sorted_eigvecs[:, : self.n_components]
        top_k_eigvals = sorted_eigvals[: self.n_components]

        # projected_data = np.dot(K_centered, self.top_k_eigvecs) sbagliato, capire perche

        return np.dot(top_k_eigvecs, np.sqrt(np.diag(top_k_eigvals)))


# a function to do kernel PCA (not a class)
import numpy as np


def my_rbfPCA(X, k, gamma):
    """
    X: numpy array
        array with n observation and p features (nxp)
    k: int
        number of principal components we want to project on
    sigma: float
        width parameter of the RBF kernel
    output: numpy array
        return the projected data (nxk), where k is the number of principal components we have projected on
    """
    # center the data in the feature space
    X_centered = X - np.mean(X, axis=0)

    # compute the squared Euclidean distances
    D = (
        np.sum(X_centered**2, axis=1).reshape(-1, 1)
        + np.sum(X_centered**2, axis=1)
        - 2 * np.dot(X_centered, X_centered.T)
    )

    # compute the RBF kernel matrix
    K = np.exp(-gamma * D)

    # double center the kernel matrix
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # perform eigen decomposition on the centered kernel matrix
    _, eigvecs = np.linalg.eigh(K_centered)

    # select the top k eigenvectors and project the original data
    top_k_eigvecs = eigvecs[:, -k:]
    projected_data = np.dot(K_centered, top_k_eigvecs)

    return projected_data


def pol(x, y, gamma, d):
    """
    Polynomial kernel function between two feature vectors.
    ----------
    Parameters:
        x, y (`numpy.ndarray`): feature vectors to compute the kernel between.
        gamma (float): Scaling factor.
        d (int): Degree of the polynomial.
    Returns:
        float: Kernel function output.
    """
    return (gamma * np.dot(x, y) + 1) ** d


class MyPolyKernelPCA:
    """
    Kernel PCA using polynomial kernel function.
    """

    def __init__(self, n_components=None, gamma=1.0, d=2):
        """
        Constructor method that initializes class variables.
        ----------
        Parameters:
        n_components: int
            Number of principal components to project on (default is None, meaning to project on all components).
        gamma: float, default = 1.0
            Scaling factor.
        d: int, default = 2
            Degree of the polynomial.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.d = d

    def fit_transform(self, X):
        """
        Compute the kernel matrix between all observations in X and use it to fit the kernel PCA model onto data X and return the projected data.
        ----------
        Parameters:
        X: numpy.ndarray
            Data to fit and transform. Required in the form: (n x p) where n = number of observations and p = number of features.
        Returns:
            numpy.ndarray: The projected data.
        """
        # compute the squared Euclidean distances
        K = np.zeros([X.shape[0], X.shape[0]])
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = pol(X[i], X[j], gamma=self.gamma, d=self.d)

        # double center the kernel matrix
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # compute the eigenvalues and eigenvectors of the kernel matrix
        eig_vals, eig_vecs = np.linalg.eig(K_centered)
        eig_vals = np.where(eig_vals < 0, 0, eig_vals)
        sorted_indices = np.argsort(eig_vals)[::-1]
        sorted_eigvecs = eig_vecs[:, sorted_indices]
        sorted_eigvals = eig_vals[sorted_indices]

        # select the first k eigenvectors and eigenvalues and project
        top_k_eigvecs = sorted_eigvecs[:, : self.n_components]
        top_k_eigvals = sorted_eigvals[: self.n_components]

        return np.dot(top_k_eigvecs, np.sqrt(np.diag(top_k_eigvals)))
