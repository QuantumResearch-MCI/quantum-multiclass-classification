import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import (
    linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
)

class KernelKNN(BaseEstimator, ClassifierMixin):
    """
    Kernel KNN klasik untuk klasifikasi multikelas.
    Jarak dihitung di feature space yang diinduksi kernel.
    """
    def __init__(
        self,
        n_neighbors=5,
        kernel='rbf',          # 'linear', 'poly', 'rbf', 'sigmoid', atau callable
        gamma=None,            # untuk rbf/poly/sigmoid
        degree=3,              # untuk poly
        coef0=1.0,             # untuk poly/sigmoid
        weights='uniform',     # 'uniform' atau 'distance'
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.weights = weights

    def _compute_kernel(self, X, Y=None):
        """Hitung kernel matrix K(X, Y)."""
        if callable(self.kernel):
            return self.kernel(X, Y if Y is not None else X)
        if self.kernel == 'linear':
            return linear_kernel(X, Y)
        if self.kernel == 'poly':
            return polynomial_kernel(X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        if self.kernel == 'rbf':
            return rbf_kernel(X, Y, gamma=self.gamma)
        if self.kernel == 'sigmoid':
            return sigmoid_kernel(X, Y, gamma=self.gamma, coef0=self.coef0)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _kernel_to_distance(self, K_xy, K_xx_diag, K_yy_diag):
        """
        d²(x, y) = K(x,x) + K(y,y) - 2K(x,y)
        K_xy: (n_x, n_y), K_xx_diag: (n_x,), K_yy_diag: (n_y,)
        """
        d2 = K_xx_diag[:, None] + K_yy_diag[None, :] - 2.0 * K_xy
        d2 = np.clip(d2, 0, None)   # antisipasi error numerik kecil
        return np.sqrt(d2)

    def fit(self, X, y):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)

        # Kernel matrix training
        K_train = self._compute_kernel(self.X_train_)
        self.K_diag_train_ = np.diag(K_train)

        # Konversi ke distance matrix
        D_train = self._kernel_to_distance(
            K_train, self.K_diag_train_, self.K_diag_train_
        )

        # KNN dengan precomputed distance
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric='precomputed',
            weights=self.weights,
        )
        self.knn_.fit(D_train, self.y_train_)
        return self

    def _compute_test_distance(self, X):
        X = np.asarray(X)
        K_test_train = self._compute_kernel(X, self.X_train_)        # (n_test, n_train)
        K_test_test  = self._compute_kernel(X, X)
        K_diag_test  = np.diag(K_test_test)
        return self._kernel_to_distance(
            K_test_train, K_diag_test, self.K_diag_train_
        )

    def predict(self, X):
        D_test = self._compute_test_distance(X)
        return self.knn_.predict(D_test)

    def predict_proba(self, X):
        D_test = self._compute_test_distance(X)
        return self.knn_.predict_proba(D_test)

    def score(self, X, y):
        return self.knn_.score(self._compute_test_distance(X), y)