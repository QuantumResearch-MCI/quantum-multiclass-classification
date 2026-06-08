# utils/quantum/qsvc_wrapper.py
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit_machine_learning.algorithms import QSVC
from .estimator import QuantumKernelEstimator

_KERNEL_PARAMS = {'n_qubits', 'lambda_', 'kernel', 'n_measurements', 'mode', 'n_features'}

class QSVCWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_qubits=4,
        lambda_=1.0,
        kernel='full',
        n_measurements=1024,
        mode='fsk',
        n_features=20,
        **qsvc_params,
    ):
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.n_measurements = n_measurements
        self.mode = mode
        self.n_features = n_features
        self.qsvc_params = qsvc_params

    def _build_model(self):
        kernel_instance = QuantumKernelEstimator(
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            n_measurements=self.n_measurements,
        )
        feature_map = kernel_instance.build_quantum_kernel(
            n_features=self.n_features,
            mode=self.mode,
        )
        return QSVC(
            quantum_kernel=feature_map,
            probability=True,
            class_weight='balanced',
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
            **self.qsvc_params,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in _KERNEL_PARAMS:
                setattr(self, key, value)
            else:
                self.qsvc_params[key] = value
        return self

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)

    def decision_function(self, X):
        return self.model_.decision_function(X)
