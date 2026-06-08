from .estimator import QuantumKernelEstimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

_KERNEL_PARAMS = {'n_qubits', 'lambda_', 'kernel', 'n_measurements', 'mode', 'n_features'}

class QXGB(BaseEstimator, ClassifierMixin):
  def __init__(
      self,
      n_qubits=8,
      lambda_=1.0,
      kernel='full',
      n_measurements=1024,
      mode='fsk',
      n_features=4,
      **xgb_params,
  ):
    self.n_qubits = n_qubits
    self.lambda_ = lambda_
    self.kernel = kernel
    self.n_measurements = n_measurements
    self.mode = mode
    self.n_features = n_features
    self.xgb_params = xgb_params

    self.binary_classifiers = {}
    self.classes_ = None
    self.X_train = None
    self.K_train = None
    self.qkernel_ = None

  def _build_model(self):
    kernel_instance = QuantumKernelEstimator(
        kernel=self.kernel,
        n_qubits=self.n_qubits,
        lambda_=self.lambda_,
        n_measurements=self.n_measurements,
    )

    self.qkernel_ = kernel_instance.build_quantum_kernel(
        n_features=self.n_features,
        mode=self.mode,
    )

    return XGBClassifier(
      objective='multi:softprob',
      tree_method='hist',
      **self.xgb_params,
    )

  def get_params(self, deep=True):
    return {
      'n_qubits': self.n_qubits,
      'lambda_': self.lambda_,
      'kernel': self.kernel,
      'n_measurements': self.n_measurements,
      'mode': self.mode,
      'n_features': self.n_features,
      **self.xgb_params,
    }

  def set_params(self, **params):
    for key, value in params.items():
      if key in _KERNEL_PARAMS:
        setattr(self, key, value)
      else:
        self.xgb_params[key] = value
    return self

  def fit(self, X, y, sample_weight=None):
    self.X_train = X
    self.classes_ = np.unique(y)
    self.model_ = self._build_model()

    K_train = self.qkernel_.evaluate(X, X)
    self.model_.fit(K_train, y, sample_weight=sample_weight)
    return self

  def predict(self, X):
    K_test = self.qkernel_.evaluate(X, self.X_train)
    return self.model_.predict(K_test)

  def predict_proba(self, X):
    K_test = self.qkernel_.evaluate(X, self.X_train)
    return self.model_.predict_proba(K_test)

  def score(self, X, y):
    K_test = self.qkernel_.evaluate(X, self.X_train)
    return self.model_.score(K_test, y)
