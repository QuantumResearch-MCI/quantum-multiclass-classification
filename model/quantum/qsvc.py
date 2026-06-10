# utils/quantum/qsvc_wrapper.py
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC
from .estimator import QuantumKernelEstimator
from .NystroemQuantumKernel import NystroemQuantumKernel

_KERNEL_PARAMS = {'n_qubits', 'lambda_', 'kernel', 'n_measurements', 'mode', 'n_features', 'gamma',
                  'm_landmarks', 'random_state'}

class QSVCWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=4,
        lambda_=1.0,
        kernel='full',
        n_measurements=1024,
        mode='fsk',
        n_features=20,
        gamma=1.0,
        m_landmarks=None,
        random_state=None,
        **qsvc_params,
    ):
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.n_measurements = n_measurements
        self.mode = mode
        self.n_features = n_features
        self.gamma = gamma
        self.m_landmarks = m_landmarks
        self.random_state = random_state
        self.qsvc_params = qsvc_params
        self.nys_ = None
        self.phi_train_ = None

    def _build_model(self):
        kernel_instance = QuantumKernelEstimator(
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            n_measurements=self.n_measurements,
            gamma=self.gamma,
        )
        qkernel = kernel_instance.build_quantum_kernel(
            n_features=self.n_features,
            mode=self.mode,
        )
        self.qkernel_ = qkernel

        if self.m_landmarks is None:
            # Native quantum-kernel SVM: kernel evaluated inside QSVC.
            return QSVC(
                quantum_kernel=qkernel,
                probability=True,
                **self.qsvc_params,
            )

        # Nyström path: spend O(N*m) circuit evals to build the explicit
        # features phi, then reconstruct the kernel Gram K ~= phi @ phi.T and
        # hand it to a precomputed-kernel SVM. The N x N Gram is only a classical
        # matmul, so keeping it full is fine; the quantum saving is already done.
        self.nys_ = NystroemQuantumKernel(
            qkernel,
            n_components=self.m_landmarks,
            random_state=self.random_state,
        )
        return SVC(
            kernel='precomputed',
            probability=True,
            **self.qsvc_params,
        )

    def get_params(self, deep=True):
        return {
            'n_qubits': self.n_qubits,
            'lambda_': self.lambda_,
            'kernel': self.kernel,
            'n_measurements': self.n_measurements,
            'mode': self.mode,
            'n_features': self.n_features,
            'gamma': self.gamma,
            'm_landmarks': self.m_landmarks,
            'random_state': self.random_state,
            **self.qsvc_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in _KERNEL_PARAMS:
                setattr(self, key, value)
            else:
                self.qsvc_params[key] = value
        return self

    def _features(self, X, fit=False):
        # Without Nyström, raw X goes straight to QSVC, which evaluates the
        # quantum kernel itself. With Nyström, return a precomputed Gram matrix:
        #   - fit:     K = phi(X) @ phi(X).T              (N x N)
        #   - predict: K = phi(X_test) @ phi(X_train).T   (N_test x N_train)
        if self.nys_ is None:
            return X
        if fit:
            self.phi_train_ = self.nys_.fit_transform(X)
            return self.phi_train_ @ self.phi_train_.T
        return self.nys_.transform(X) @ self.phi_train_.T

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(self._features(X, fit=True), y)
        return self

    def predict(self, X):
        return self.model_.predict(self._features(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._features(X))

    def score(self, X, y):
        return self.model_.score(self._features(X), y)

    def decision_function(self, X):
        return self.model_.decision_function(self._features(X))
