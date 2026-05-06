# utils/quantum/qsvc_wrapper.py
from sklearn.base import BaseEstimator, ClassifierMixin
from qiskit_machine_learning.algorithms import QSVC
from .estimator import QuantumKernelEstimator

class QSVCWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        kernel, 
        n_qubits=4,
        lambda_=1.0, 
        C=1.0, 
        class_weight=None, 
        decision_function_shape='ovr', 
        n_measurements=1024, 
        use_hardware=False, 
        n_features=20, 
        random_state=42
      ):
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.C = C
        self.n_measurements = n_measurements
        self.use_hardware = use_hardware
        self.n_features = n_features
        self.random_state = random_state
        self.class_weight = class_weight
        self.decision_function_shape = decision_function_shape

    def _build_model(self):
        kernel_instance = QuantumKernelEstimator(
            kernel=self.kernel,
            n_qubits=self.n_qubits,
            lambda_=self.lambda_,
            n_measurements=self.n_measurements,
        )
        feature_map = kernel_instance.build_quantum_kernel(
            n_features=self.n_features,
            use_hardware=self.use_hardware,
        )
        return QSVC(
          quantum_kernel=feature_map, 
          probability=True, 
          C=self.C, 
          random_state=self.random_state, 
          class_weight=self.class_weight,
          decision_function_shape=self.decision_function_shape
        )

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