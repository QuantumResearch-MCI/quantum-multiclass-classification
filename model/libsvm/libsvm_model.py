from libsvm.svmutil import svm_train, svm_predict
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class LibSVMModel(BaseEstimator, ClassifierMixin):
    def __init__(self, kernel='rbf', C=1.0, gamma=None, degree=None):
        self.kernel_map = {'linear': 0, 'poly': 1, 'rbf': 2, 'sigmoid': 3}
        self.t_val = self.kernel_map[kernel]
        self.params = f"-t {self.t_val} -c {C} -b 1 -q"

        if kernel == 'linear':
            pass
        elif kernel == 'poly':
            self.params += f" -g {gamma} -d {degree}"
        elif kernel in ['rbf', 'sigmoid']:
            self.params += f" -g {gamma}"

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