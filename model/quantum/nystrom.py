"""
Nyström approximation for quantum kernels.

Implements the kernel matrix approximation described in:
    Ghojogh et al. (2021), "Reproducing Kernel Hilbert Space, Mercer's Theorem,
    Eigenfunctions, Nyström Method, and Use of Kernels in Machine Learning",
    Section 10.3 (Equations 130-137).

The full n×n kernel matrix K is approximated from m << n landmark points:

          [ A   B ]         [ A    B        ]
    K  =  [       ]    ≈    [               ] ,    C ≈ Bᵀ A⁺ B
          [ Bᵀ  C ]         [ Bᵀ   Bᵀ A⁺ B  ]

where:
    A = K(L, L)  ∈ ℝ^(m×m)   — landmark vs landmark kernel
    B = K(L, X)  ∈ ℝ^(m×n)   — landmark vs all-data kernel

For quantum kernels this is critical: each entry K(x_i, x_j) requires
a separate circuit execution, so reducing eval count from
O(n²) → O(m·n + m²) is the difference between hours and minutes.

Supports both `fsk` (FidelityStatevectorKernel) and `fqk`
(FidelityQuantumKernel) modes transparently via QuantumKernelEstimator.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from .estimator import QuantumKernelEstimator


class QuantumNystromKernel(BaseEstimator, TransformerMixin):
    """
    Nyström approximator wrapping QuantumKernelEstimator.

    Parameters
    ----------
    n_qubits, lambda_, kernel, n_measurements, mode, n_features
        Passed through to QuantumKernelEstimator — identical semantics
        to QSVCWrapper / QESVC / QXGB.
    n_landmarks : int
        Number of landmark points m. Larger = more accurate but more
        circuit calls. Rule of thumb: m ≈ 4·√n is a sane starting point.
    landmark_method : {'random', 'kmeans'}
        - 'random' : standard Nyström, fast, no extra preprocessing.
        - 'kmeans' : picks the training point closest to each k-means
          centroid (Zhang & Kwok 2010). Slower to set up but typically
          tighter approximation, especially when n_landmarks is small.
    regularization : float
        Tikhonov shift added to A's eigenvalues before pseudo-inversion.
        Guards against ill-conditioning of A.
    random_state : int
        Seed for landmark sampling and KMeans.
    verbose : bool
        Print how many quantum kernel evaluations Nyström saves.
    """

    def __init__(
        self,
        n_qubits=4,
        lambda_=1.0,
        kernel='full',
        n_measurements=1024,
        mode='fsk',
        n_features=4,
        n_landmarks=100,
        landmark_method='random',
        regularization=1e-6,
        random_state=42,
        verbose=True,
    ):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.kernel = kernel
        self.n_measurements = n_measurements
        self.mode = mode
        self.n_features = n_features
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.regularization = regularization
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Landmark selection
    # ------------------------------------------------------------------
    def _select_landmarks(self, X):
        n = X.shape[0]
        m = min(self.n_landmarks, n)

        if self.landmark_method == 'random':
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n, size=m, replace=False)
            return X[idx], idx

        if self.landmark_method == 'kmeans':
            km = KMeans(
                n_clusters=m,
                random_state=self.random_state,
                n_init=10,
            ).fit(X)
            # Project centroids back to the nearest actual training points
            # (Nyström needs landmarks to be members of X)
            idx, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
            idx = np.unique(idx)  # rare dedup
            return X[idx], idx

        raise ValueError(
            f"Unknown landmark_method='{self.landmark_method}'. "
            "Choose 'random' or 'kmeans'."
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        """
        Build the Nyström approximation in three steps:
          1. Select m landmarks L from X.
          2. Compute A = K(L, L)  and  B = K(L, X)  via quantum kernel.
          3. Compute regularized pseudo-inverse A⁺ via eigendecomp.

        Total quantum kernel evaluations: m² + m·n   (vs n² for full).
        """
        X = np.asarray(X)
        n = X.shape[0]

        # --- Build underlying quantum kernel (works for fsk and fqk) ---
        self._kernel_estimator = QuantumKernelEstimator(
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            n_measurements=self.n_measurements,
        )
        self._qkernel = self._kernel_estimator.build_quantum_kernel(
            n_features=self.n_features,
            mode=self.mode,
        )

        # --- 1. Pick landmarks ---
        self.landmarks_, self.landmark_idx_ = self._select_landmarks(X)
        m = self.landmarks_.shape[0]

        if self.verbose:
            full_evals = n * (n + 1) // 2
            nys_evals = m * (m + 1) // 2 + m * (n - m)
            speedup = full_evals / max(nys_evals, 1)
            print(f"[Nyström] mode={self.mode}, n={n}, m={m}")
            print(f"[Nyström] quantum kernel evaluations: "
                  f"{nys_evals:,}  (full would be {full_evals:,}; "
                  f"{speedup:.1f}× reduction)")

        # --- 2a. Compute A = K(L, L), enforce symmetry ---
        A = self._qkernel.evaluate(self.landmarks_, self.landmarks_)
        A = (A + A.T) / 2.0

        # --- 3. Regularized pseudo-inverse via eigendecomposition ---
        # A is PSD by construction; eigh is the stable choice.
        eigvals, eigvecs = np.linalg.eigh(A)
        max_eig = max(eigvals.max(), 0.0)
        threshold = max(self.regularization, max_eig * 1e-10)
        keep = eigvals > threshold
        eigvals_safe = np.where(keep, eigvals, threshold)

        inv_diag = np.where(keep, 1.0 / eigvals_safe, 0.0)
        inv_sqrt_diag = np.where(keep, 1.0 / np.sqrt(eigvals_safe), 0.0)
        self.A_pinv_ = (eigvecs * inv_diag) @ eigvecs.T
        self.A_inv_sqrt_ = (eigvecs * inv_sqrt_diag) @ eigvecs.T

        # --- 2b. Compute B = K(L, X_train), cache for inference ---
        self.X_train_ = X
        self.B_train_ = self._qkernel.evaluate(self.landmarks_, X)

        return self

    def transform(self, X):
        """
        Nyström-approximated kernel matrix between new points X and training data:
            K(X, X_train)  ≈  K(X, L) · A⁺ · K(L, X_train)
        Shape: (n_X, n_train). Use this as the test kernel for
        SVC(kernel='precomputed').
        """
        X = np.asarray(X)
        K_XL = self._qkernel.evaluate(X, self.landmarks_)   # n_X × m
        return K_XL @ self.A_pinv_ @ self.B_train_

    def fit_transform(self, X, y=None):
        """
        Fit on X, then return approximated K(X_train, X_train) ≈ Bᵀ A⁺ B.
        Faster than fit(X).transform(X) because B is already cached.
        """
        self.fit(X, y)
        return self.B_train_.T @ self.A_pinv_ @ self.B_train_

    def transform_features(self, X):
        """
        Explicit Nyström feature map  φ(X) = K(X, L) · A^(-1/2)  ∈  ℝ^(n × m).
        Satisfies  K(x, y) ≈ φ(x) · φ(y)ᵀ.

        Useful if you want to feed an explicit low-dim quantum embedding
        into a non-kernel model (XGBoost, MLP, logistic regression, …).
        """
        X = np.asarray(X)
        K_XL = self._qkernel.evaluate(X, self.landmarks_)
        return K_XL @ self.A_inv_sqrt_