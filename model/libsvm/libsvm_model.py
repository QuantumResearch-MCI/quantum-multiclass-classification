from libsvm.svmutil import svm_train, svm_predict
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class LibSVMModel(BaseEstimator, ClassifierMixin):
  def __init__(self, params):
      self.params = params
      self.model = None

  def fit(self, X, y):
      self.model_ = svm_train(y.tolist(), X.tolist(), self.params)
      self.classes_ = np.unique(y)
      return self

  def predict(self, X, y=None):
      if y is None:
          y = [0] * len(X)  # dummy labels required by svm_predict
      preds, _, _ = svm_predict(y if not hasattr(y, 'tolist') else y.tolist(), X.tolist(), self.model_, '-q')
      return preds
  
  def predict_proba(self, X):
      y_dummy = [0] * len(X)
      _, _, prob_estimates = svm_predict(
          y_dummy, X.tolist(), self.model_, '-b 1 -q'
      )
      return np.array(prob_estimates)