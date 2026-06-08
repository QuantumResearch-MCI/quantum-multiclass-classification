"""
QESVM with Nyström-approximated quantum kernel.

Inherits from QESVM and only overrides the kernel-computation hooks so
the OvR training loop, decision function, and prediction code stay
identical to the paper's Algorithm 1 — only the kernel matrix itself is
approximated.

Quantum kernel evaluations:
    Plain QESVM    :  O(n²)   circuit calls for training
                      O(n·n_test) for inference
    QESVM-Nyström  :  O(m² + m·n)  for training
                      O(m·n_test)  for inference
where m = n_landmarks ≪ n.
"""

import numpy as np

from .qesvc import QESVM
from .nystrom import QuantumNystromKernel


class QESVMNystrom(QESVM):
    """
    Drop-in subclass of QESVM that swaps the exact quantum kernel for a
    Nyström approximation. Inherits all paper-faithful OvR training,
    decision function, and prediction logic from QESVM.

    Extra parameters
    ----------------
    n_landmarks : int
        Number of landmark points m. Try m ≈ 4·√n as a starting point.
    landmark_method : {'random', 'kmeans'}
        Landmark selection strategy. 'kmeans' projects centroids back to
        nearest training points (Zhang & Kwok 2010).
    regularization : float
        Tikhonov shift on A's eigenvalues, plus a tiny ridge on K_train
        for SVC's dual stability.
    """

    def __init__(
        self,
        # — Quantum + multiclass params (forwarded to QESVM) —
        n_qubits=4,
        lambda_=1.0,
        kernel='full',
        n_measurements=1024,
        mode='fsk',
        n_features=4,
        C=1.0,
        random_state=42,
        class_weight=None,
        tol=1e-3,
        verbose=False,
        # — Nyström params —
        n_landmarks=100,
        landmark_method='random',
        regularization=1e-6,
    ):
        super().__init__(
            n_qubits=n_qubits,
            lambda_=lambda_,
            kernel=kernel,
            n_measurements=n_measurements,
            mode=mode,
            n_features=n_features,
            C=C,
            random_state=random_state,
            class_weight=class_weight,
            tol=tol,
            verbose=verbose,
        )
        self.n_landmarks = n_landmarks
        self.landmark_method = landmark_method
        self.regularization = regularization

    # ------------------------------------------------------------------
    # Override the three kernel hooks — everything else stays in QESVM
    # ------------------------------------------------------------------
    def _setup_kernel(self):
        """
        Construct the Nyström wrapper. We do NOT eagerly build the inner
        quantum kernel here — QuantumNystromKernel.fit() does that, and
        importantly only computes A (m×m) and B (m×n) rather than the
        full n×n matrix.
        """
        self.nystrom_ = QuantumNystromKernel(
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            kernel=self.kernel,
            n_measurements=self.n_measurements,
            mode=self.mode,
            n_features=self.n_features,
            n_landmarks=self.n_landmarks,
            landmark_method=self.landmark_method,
            regularization=self.regularization,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    def _compute_train_kernel(self, X):
        """
        Compute the Nyström-approximated K(X, X):
            K ≈ Bᵀ · A⁺ · B
        with symmetrization and a small PSD ridge to keep SVC's dual
        well-posed even when A is near-rank-deficient.
        """
        K = self.nystrom_.fit_transform(X)
        K = (K + K.T) / 2.0
        K += self.regularization * np.eye(K.shape[0])
        return K

    def _compute_test_kernel(self, X):
        """
        K(X_test, X_train) ≈ K(X_test, L) · A⁺ · K(L, X_train),
        reusing the landmarks chosen at fit time.
        """
        return self.nystrom_.transform(np.asarray(X))