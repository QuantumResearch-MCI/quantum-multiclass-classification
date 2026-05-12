from .estimator import QuantumKernelEstimator
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from catboost import CatBoostClassifier

class QCAT(BaseEstimator, ClassifierMixin):
  def __init__(
      self,
      n_qubits=8, 
      lambda_=1.0, 
      kernel='full', 
      n_measurements=1024, 
      use_hardware=False, 
      n_features=4,
      
      random_seed=42,
      iterations=100,
      depth=5,
      learning_rate=0.1,
      l2_leaf_reg=3,
      random_strength=1,
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
    self.random_seed = random_seed
    self.iterations = iterations
    self.depth = depth
    self.l2_leaf_reg = l2_leaf_reg
    self.learning_rate = learning_rate
    self.random_strength = random_strength
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

    return CatBoostClassifier(
      loss_function="MultiClassOneVsAll",
      iterations=self.iterations,
      learning_rate=self.learning_rate,
      depth=self.depth,
      eval_metric="Accuracy",
      random_seed=self.random_seed,
      verbose=0,
      l2_leaf_reg=self.l2_leaf_reg,
      random_strength=self.random_strength,
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
  
