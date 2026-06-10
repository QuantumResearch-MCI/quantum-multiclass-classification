"""
Nyström approximation for quantum kernels.

Replaces the full N x N kernel matrix with an explicit N x m feature map

    phi(x) = K(x, L) @ K(L, L)^(-1/2)

where L is a set of m << N landmark points sampled from the training data.
Any linear model (or tree model treating the columns as features) can then
consume `phi` directly, since K(X, X) ~= phi(X) @ phi(X).T.

Why it helps
------------
For fidelity-based kernels ('fqk', 'fsk', 'fqk-hardware') the full Gram matrix
costs O(N^2) circuit evaluations. Nyström only needs:
    - K(L, L)  -> O(m^2) evaluations
    - K(X, L)  -> O(N * m) evaluations
i.e. O(N * m) total, a large saving when m << N.

For the Projected Quantum Kernel the quantum cost is already O(N) (one
projection per sample), so Nyström does not reduce the quantum cost there;
it only shrinks the classical RBF Gram from O(N^2) to O(N * m) and yields
explicit m-dimensional features.

The API mirrors a minimal sklearn transformer (fit / transform / fit_transform)
but, unlike sklearn.kernel_approximation.Nystroem, it calls `.evaluate` on whole
matrices at once instead of per-pair, so quantum circuit batching is preserved.

Reference:
    Williams, C. K. I., & Seeger, M. (2001). Using the Nyström method to speed
    up kernel machines. NeurIPS 13.
"""

import numpy as np


class NystroemQuantumKernel:
    """
    Nyström feature approximation wrapping any kernel with `.evaluate(X, Y)`.

    Parameters
    ----------
    qkernel : object
        A built quantum kernel exposing `evaluate(X, Y)` (e.g. the object
        returned by QuantumKernelEstimator.build_quantum_kernel). Works with
        FidelityQuantumKernel, FidelityStatevectorKernel and
        ProjectedQuantumKernel.
    n_components : int, default=100
        Number of landmark points m. Clipped to the number of training samples.
    random_state : int or None, default=None
        Seed for landmark sampling (reproducible folds / HPO runs).
    reg : float, default=1e-12
        Floor applied to the eigenvalues of K(L, L) before inverse-sqrt, for
        numerical stability when the landmark Gram is near-singular.

    Attributes
    ----------
    landmarks_ : ndarray of shape (m, n_features)
        Sampled landmark inputs.
    normalization_ : ndarray of shape (m, m)
        K(L, L)^(-1/2), computed via symmetric eigendecomposition.

    Examples
    --------
    >>> nys = NystroemQuantumKernel(qkernel, n_components=80, random_state=0)
    >>> Phi_train = nys.fit_transform(X_train)   # (N, m)
    >>> Phi_test = nys.transform(X_test)         # (N_test, m)
    """

    def __init__(self, qkernel, n_components=100, random_state=None, reg=1e-12):
        self.qkernel = qkernel
        self.n_components = n_components
        self.random_state = random_state
        self.reg = reg

    def fit(self, X, y=None):
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        rng = np.random.default_rng(self.random_state)

        n_samples = X.shape[0]
        m = min(self.n_components, n_samples)
        idx = rng.choice(n_samples, size=m, replace=False)
        self.landmarks_ = X[idx]

        # K(L, L): m x m landmark Gram matrix.
        K_mm = np.asarray(self.qkernel.evaluate(self.landmarks_, self.landmarks_))

        # K(L, L)^(-1/2) via symmetric eigendecomposition. K_mm is PSD by
        # construction; clip eigenvalues to `reg` to stay invertible.
        eigvals, eigvecs = np.linalg.eigh(K_mm)
        eigvals = np.maximum(eigvals, self.reg)
        self.normalization_ = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        return self

    def transform(self, X):
        if not hasattr(self, 'normalization_'):
            raise RuntimeError("NystroemQuantumKernel must be fit before transform().")
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        # K(X, L): N x m, then project into the Nyström feature space.
        K_nm = np.asarray(self.qkernel.evaluate(X, self.landmarks_))
        return K_nm @ self.normalization_

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
