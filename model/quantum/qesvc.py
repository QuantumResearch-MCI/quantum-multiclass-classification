"""
Quantum-Enhanced Multiclass SVM (QESVM).

Faithful implementation of Algorithm 1 from:
    Ding, Wang, Wang, Gao (2024),
    "Quantum machine learning for multiclass classification beyond kernel methods",
    arXiv:2411.02913.

The paper's algorithm follows One-vs-Rest (OvR) explicitly:

    Stage 1 — Quantum kernel estimation:
        K_ij = |<φ(x_i) | φ(x_j)>|²

    Stage 2 — Train l binary SVMs, one per class:
        for s in {1, …, l}:
            L_s[i] = +1 if y_i == s else -1
            (α^s, b_s) = SMO(K, L_s, C)

    Prediction:
        f_s(x_t) = Σ_{i ∈ Ω} α_i^s y_i K_it + b_s         (Algorithm 1, line 35)
        ỹ        = argmax_s f_s(x_t)                       (Algorithm 1, line 36)

This differs from the existing `QESVC` wrapper which delegates to a single
`sklearn.SVC(kernel=qkernel.evaluate)` — SVC internally uses One-vs-One, not OvR.
QESVM trains l independent binary classifiers explicitly on a precomputed K,
matching the paper exactly.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

from .estimator import QuantumKernelEstimator


class QESVM(BaseEstimator, ClassifierMixin):
    """
    Quantum-Enhanced multiclass SVM (one-vs-rest).

    Parameters
    ----------
    n_qubits, lambda_, kernel, n_measurements, mode, n_features
        Forwarded to QuantumKernelEstimator. `mode` ∈ {'fsk', 'fqk', 'fqk-hardware'}.
    C : float
        Soft-margin penalty for each binary SVM (paper Eq. 24).
    random_state, class_weight, tol
        Standard sklearn SVC controls.
    verbose : bool
        Print progress during quantum kernel computation and binary training.
    """

    def __init__(
        self,
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
    ):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.kernel = kernel
        self.n_measurements = n_measurements
        self.mode = mode
        self.n_features = n_features
        self.C = C
        self.random_state = random_state
        self.class_weight = class_weight
        self.tol = tol
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Hooks — subclasses (e.g. Nyström) override these
    # ------------------------------------------------------------------
    def _setup_kernel(self):
        """Build the underlying quantum kernel."""
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

    def _compute_train_kernel(self, X):
        """K(X, X) for training. Override for approximations."""
        K = self._qkernel.evaluate(X, X)
        return (K + K.T) / 2.0

    def _compute_test_kernel(self, X):
        """K(X_test, X_train). Override for approximations."""
        return self._qkernel.evaluate(np.asarray(X), self.X_train_)

    # ------------------------------------------------------------------
    # Stage 1 + Stage 2: kernel estimation + l binary SVMs
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Stage 1 — Quantum kernel estimation
        self._setup_kernel()
        if self.verbose:
            print(f"[QESVM] Stage 1: computing K ({X.shape[0]}×{X.shape[0]}) "
                  f"with mode={self.mode}, kernel={self.kernel}")
        K_train = self._compute_train_kernel(X)

        self.X_train_ = X
        self.classes_ = np.unique(y)
        l = len(self.classes_)

        # Stage 2 — Train l binary SVMs (one-vs-rest), Algorithm 1 lines 22-32
        if self.verbose:
            print(f"[QESVM] Stage 2: training {l} binary SVMs (OvR)")

        self.binary_classifiers_ = {}
        for s_idx, cls in enumerate(self.classes_):
            # Construct binary label vector L_s  (Algorithm 1 lines 24-29)
            L_s = np.where(y == cls, 1, -1)

            # Solve binary SVM via SMO on precomputed K  (Algorithm 1 line 31)
            clf = SVC(
                kernel='precomputed',
                C=self.C,
                random_state=self.random_state,
                class_weight=self.class_weight,
                tol=self.tol,
                probability=True,
            )
            clf.fit(K_train, L_s)
            self.binary_classifiers_[cls] = clf

            if self.verbose:
                n_sv = int(clf.n_support_.sum())
                print(f"[QESVM]   class={cls}: {n_sv} support vectors")

        return self

    # ------------------------------------------------------------------
    # Prediction — Algorithm 1 lines 34-36
    # ------------------------------------------------------------------
    def decision_function(self, X):
        """
        Returns matrix of shape (n_samples, n_classes), where column s is
            f_s(x) = Σ_{i ∈ Ω} α_i^s y_i K(x, x_i) + b_s
        as in Eq. (10) / Algorithm 1 line 35.
        """
        K_test = self._compute_test_kernel(X)
        return self._decision_from_kernel(K_test)

    def _decision_from_kernel(self, K_test):
        decisions = np.zeros((K_test.shape[0], len(self.classes_)))
        for s_idx, cls in enumerate(self.classes_):
            decisions[:, s_idx] = self.binary_classifiers_[cls].decision_function(K_test)
        return decisions

    def predict(self, X):
        """ỹ = argmax_s f_s(x)  (Algorithm 1 line 36)."""
        decisions = self.decision_function(X)
        return self.classes_[np.argmax(decisions, axis=1)]

    def predict_proba(self, X):
        """
        Per-class probability via Platt-scaled binary SVMs, normalized to
        sum to 1 across classes (OvR convention — sklearn's standard trick).
        """
        K_test = self._compute_test_kernel(X)
        probs = np.zeros((K_test.shape[0], len(self.classes_)))
        for s_idx, cls in enumerate(self.classes_):
            # SVC.predict_proba returns [[P(-1), P(+1)], …]; P(+1) = P(this class)
            p = self.binary_classifiers_[cls].predict_proba(K_test)
            probs[:, s_idx] = p[:, 1]
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))