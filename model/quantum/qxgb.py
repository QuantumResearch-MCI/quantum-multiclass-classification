from .estimator import QuantumKernelEstimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

class QXGB(BaseEstimator, ClassifierMixin):
  def __init__(
      self,
      n_qubits=8, 
      lambda_=1.0, 
      kernel='full', 
      n_measurements=1024, 
      use_hardware=False, 
      n_features=4,
      random_state=42,
      n_estimators=500,
      max_depth=10,
      subsample=0.8,
      learning_rate=0.5,
      booster='gbtree',
  ):
    self.n_qubits = n_qubits
    self.lambda_ = lambda_
    self.kernel = kernel
    self.n_measurements = n_measurements        
    self.binary_classifiers = {}
    self.classes_ = None
    self.X_train = None
    self.K_train = None
    self.use_hardware = use_hardware
    self.n_features = n_features
    self.random_state = random_state
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.subsample = subsample
    self.learning_rate = learning_rate
    self.booster = booster
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
        use_hardware=self.use_hardware,
    )

    return XGBClassifier(
      booster=self.booster,
      objective = 'multi:softprob',
      random_state=self.random_state,
      n_estimators=self.n_estimators,
      max_depth=self.max_depth,
      subsample=self.subsample,
      learning_rate=self.learning_rate,
    )
  
  def fit(self, X, y):
    self.X_train = X
    self.classes_ = np.unique(y)
    self.model_ = self._build_model()

    K_train = self.qkernel_.evaluate(X, X)
    self.model_.fit(K_train, y)
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
  
