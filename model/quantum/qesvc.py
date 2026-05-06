from .estimator import QuantumKernelEstimator
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class QESVC(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            n_qubits=8, 
            lambda_=1.0, 
            kernel='full', 
            C=1.0, 
            n_measurements=1024, 
            use_hardware=False, 
            n_features=4,
            random_state=42,
            class_weight=None,
            shrinking=False,
            tol=1,
            decision_function_shape='ovr'
        ):
        self.n_qubits = n_qubits
        self.lambda_ = lambda_
        self.kernel = kernel
        self.C = C
        self.n_measurements = n_measurements        
        self.binary_classifiers = {}
        self.classes_ = None
        self.X_train = None
        self.K_train = None
        self.use_hardware = use_hardware
        self.n_features = n_features
        self.random_state = random_state
        self.class_weight = class_weight
        self.shrinking = shrinking
        self.tol = tol
        self.decision_function_shape = decision_function_shape
        
    def fit(self, X, y):
        self.kernel_estimator = QuantumKernelEstimator(
            kernel=self.kernel, 
            n_qubits=self.n_qubits, 
            lambda_=self.lambda_, 
            n_measurements=self.n_measurements
        )

        self.kernel_estimator.build_quantum_kernel(
            n_features=self.n_features, 
            use_hardware=self.use_hardware
        )
        
        self.X_train = X
        self.classes_ = np.unique(y)

        # Step 1: Compute kernel matrix for training data
        qkernel = self.kernel_estimator._qkernel
        
        # Step 2: Train binary classifiers for each class (one-vs-all)
        for s in self.classes_:
            y_binary = np.where(y == s, 1, -1)
            clf = SVC(
                kernel=qkernel.evaluate, 
                C=self.C, 
                random_state=self.random_state,
                class_weight=self.class_weight,
                shrinking=self.shrinking,
                tol=self.tol,
                decision_function_shape=self.decision_function_shape
            )
            clf.fit(X, y_binary)
            self.binary_classifiers[s] = clf
            
        return self
    
    def decision_function(self, X_test):
        n_test = X_test.shape[0]
        decision_values = np.zeros((n_test, len(self.classes_)))
        
        # Calculate decision function for each class
        for idx, s in enumerate(self.classes_):
            decision_values[:, idx] = self.binary_classifiers[s].decision_function(X_test)
            
        return decision_values
    
    def predict(self, X_test):
        decision_values = self.decision_function(X_test)
        predictions = self.classes_[np.argmax(decision_values, axis=1)]
        return predictions
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy