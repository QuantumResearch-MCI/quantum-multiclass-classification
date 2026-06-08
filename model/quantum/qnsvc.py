# model/quantum/qnsvc.py
# Nystrom Quantum Kernel Estimation (NQKE) wrapper, following
# Srikumar, Hill & Hollenberg, "A kernel-based quantum random forest
# for improved classification" (arXiv:2210.02355), Appendix A4 / Eq.(A45),(A49).
#
# Idea: instead of computing the full N x N quantum Gram matrix (O(N^2) kernel
# evaluations), randomly pick L landmark points and compute only the N x L
# columns (O(N*L) evaluations). Reconstruct an explicit Nystrom feature map
#   N_W(x)_i = sum_j k(x, z_j) (W^{-1/2})_{ij}
# and fit a *linear* SVC on top of it. This is the single-SVM (QSVC) use of
# NQKE -- not the full tree/forest.

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from .estimator import QuantumKernelEstimator


class QNSVC(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=8,
        lambda_=1.0,
        kernel='full',
        n_measurements=1024,
        mode='fsk',
        n_features=4,

        # Nystrom-specific
        n_landmarks=10,      # L in the paper
        rank=None,           # rank-r truncation of W (None -> keep all positive eigvals)
        eig_tol=1e-10,       # drop eigenvalues below this before pseudo-inverse

        # SVC-specific
        C=1.0,
        random_state=42,
        class_weight='balanced',
        decision_function_shape='ovr',
        tol=1e-3,
    ):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.kernel = kernel
        self.n_measurements = n_measurements
        self.mode = mode
        self.n_features = n_features

        self.n_landmarks = n_landmarks
        self.rank = rank
        self.eig_tol = eig_tol

        self.C = C
        self.random_state = random_state
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape
        self.tol = tol

    # ------------------------------------------------------------------
    def _build_kernel(self):
        est = QuantumKernelEstimator(
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            n_measurements=self.n_measurements,
        )
        est.build_quantum_kernel(n_features=self.n_features, mode=self.mode)
        return est

    def _compute_W_inv_sqrt(self, W):
        # Symmetric eigendecomposition; W is PSD (quantum kernel) but may be
        # singular -> use a rank-truncated Moore-Penrose-style inverse sqrt.
        eigvals, eigvecs = np.linalg.eigh(W)

        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        if self.rank is not None:
            eigvals = eigvals[:self.rank]
            eigvecs = eigvecs[:, :self.rank]

        keep = eigvals > self.eig_tol
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

        inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        return inv_sqrt

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        rng = np.random.default_rng(self.random_state)
        N = X.shape[0]
        L = min(self.n_landmarks, N)

        # NOTE: quantum kernels have K_ii = 1, so uniform landmark sampling is
        # equivalent to diagonal-weighted sampling (paper, Corollary 1).
        idx = rng.choice(N, size=L, replace=False)
        self.landmarks_ = X[idx]

        self.est_ = self._build_kernel()

        # W = K(landmarks, landmarks)  -> L x L   (only L^2 evaluations)
        W = self.est_.compute_kernel_matrix(self.landmarks_)        # L x L
        self.W_inv_sqrt_ = self._compute_W_inv_sqrt(W)

        # Explicit Nystrom features for the training set (the N x L columns)
        X_feat = self._transform(X)

        self.model_ = SVC(
            kernel='rbf',
            C=self.C,
            probability=True,
            random_state=self.random_state,
            class_weight=self.class_weight,
            decision_function_shape=self.decision_function_shape,
            tol=self.tol,
        )
        self.model_.fit(X_feat, y)
        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        # v = [k(x, z_1), ..., k(x, z_L)] for every x  -> shape (n, L)
        V = self.est_.compute_kernel_matrix(X, self.landmarks_)     # n x L
        return V @ self.W_inv_sqrt_                                 # Eq.(A49)

    # ------------------------------------------------------------------
    def predict(self, X):
        return self.model_.predict(self._transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._transform(X))

    def decision_function(self, X):
        return self.model_.decision_function(self._transform(X))

    def score(self, X, y):
        return self.model_.score(self._transform(X), np.asarray(y))