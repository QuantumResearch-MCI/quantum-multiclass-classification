from .estimator import QuantumKernelEstimator
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class QESVC(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            n_qubits=8, 
            lambda_=1.0, 
            kernel_type='full', 
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
        self.kernel_type = kernel_type
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
            kernel_type=self.kernel_type, 
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

        # Step 1: Compute quantum kernel matrix
        self.K_train = self.kernel_estimator.compute_kernel_matrix(X)
        
        # Step 2: Train binary classifiers for each class (one-vs-all)
        self.binary_classifiers_ = {}
        for s in self.classes_:
            y_binary = np.where(y == s, 1, -1)
            clf = SVC(
                kernel='precomputed', 
                C=self.C, 
                random_state=self.random_state,
                class_weight=self.class_weight,
                shrinking=self.shrinking,
                tol=self.tol,
                decision_function_shape=self.decision_function_shape
            )
            clf.fit(self.K_train, y_binary)
            self.binary_classifiers[s] = clf
            
        return self
    
    def decision_function(self, X_test):
        n_test = X_test.shape[0]
        decision_values = np.zeros((n_test, len(self.classes_)))
        
        # Compute kernel between test and training data
        K_test = self.kernel_estimator.compute_kernel_matrix(X_test, self.X_train)
        
        # Calculate decision function for each class
        for idx, s in enumerate(self.classes_):
            decision_values[:, idx] = clf = self.binary_classifiers[s].decision_function(K_test)
            
        return decision_values
    
    def predict(self, X_test):
        decision_values = self.decision_function(X_test)
        predictions = self.classes_[np.argmax(decision_values, axis=1)]
        return predictions
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        return accuracy