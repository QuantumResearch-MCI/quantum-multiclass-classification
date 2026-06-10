from .estimator import QuantumKernelEstimator
from .NystroemQuantumKernel import NystroemQuantumKernel
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

_KERNEL_PARAMS = {'n_qubits', 'lambda_', 'kernel', 'n_measurements', 'mode', 'n_features', 'gamma',
                  'm_landmarks', 'random_state'}

class QCAT(BaseEstimator, ClassifierMixin):
  def __init__(
      self,
      n_qubits=8,
      lambda_=1.0,
      kernel='full',
      n_measurements=1024,
      mode='fsk',
      n_features=4,
      gamma=1.0,
      m_landmarks=None,
      random_state=None,
      **cat_params,
  ):
    self.n_qubits = n_qubits
    self.lambda_ = lambda_
    self.kernel = kernel
    self.n_measurements = n_measurements
    self.mode = mode
    self.n_features = n_features
    self.gamma = gamma
    self.m_landmarks = m_landmarks
    self.random_state = random_state
    self.cat_params = cat_params

    self.binary_classifiers = {}
    self.classes_ = None
    self.X_train = None
    self.K_train = None
    self.qkernel_ = None
    self.nys_ = None

  def _build_model(self):
    kernel_instance = QuantumKernelEstimator(
        kernel=self.kernel,
        n_qubits=self.n_qubits,
        lambda_=self.lambda_,
        n_measurements=self.n_measurements,
        gamma=self.gamma,
    )

    self.qkernel_ = kernel_instance.build_quantum_kernel(
        n_features=self.n_features,
        mode=self.mode,
    )

    return CatBoostClassifier(
      devices='GPU',
      loss_function="MultiClassOneVsAll",
      eval_metric="Accuracy",
      verbose=0,
      **self.cat_params,
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
      **self.cat_params,
    }

  def set_params(self, **params):
    for key, value in params.items():
      if key in _KERNEL_PARAMS:
        setattr(self, key, value)
      else:
        self.cat_params[key] = value
    return self

  def _features(self, X, fit=False):
    # With m_landmarks set, use Nyström features (N x m); otherwise fall back
    # to the full kernel matrix (N x N) against the training set.
    if self.m_landmarks is None:
      return self.qkernel_.evaluate(X, self.X_train)
    if fit:
      self.nys_ = NystroemQuantumKernel(
          self.qkernel_,
          n_components=self.m_landmarks,
          random_state=self.random_state,
      ).fit(X)
    return self.nys_.transform(X)

  def fit(self, X, y):
    self.X_train = X
    self.classes_ = np.unique(y)
    self.model_ = self._build_model()

    Phi_train = self._features(X, fit=True)
    self.model_.fit(Phi_train, y)
    return self

  def predict(self, X):
    return self.model_.predict(self._features(X))

  def predict_proba(self, X):
    return self.model_.predict_proba(self._features(X))

  def score(self, X, y):
    return self.model_.score(self._features(X), y)
