def svc_model(kernel, random_state=None):
    from sklearn.svm import SVC
    model = SVC(kernel=kernel, random_state=random_state, probability=True)
    return model

def xgboost_model(booster='gbtree', n_estimators=500, max_depth=10, subsample=0.8, learning_rate=1, random_state=42):
    from xgboost import XGBClassifier
    model = XGBClassifier(
                # n_estimators=n_estimators, 
                # max_depth=max_depth, 
                # subsample=subsample, 
                # learning_rate=learning_rate, 
                booster=booster,
                objective='multi:softprob', 
                random_state=random_state,
            )
    return model

def catboost_model(iterations=1000, learning_rate=0.1, depth=6, random_seed=42):
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
                loss_function="MultiClassOneVsAll",
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                eval_metric="Accuracy",
                verbose=0,
                random_seed=random_seed,
            )
    return model

def qsvc_model(kernel, random_state=42, use_hardware=False, n_features=20, decision_function_shape='ovr', n_qubits=4):
    # from qiskit_machine_learning.algorithms import QSVC
    # from .quantum.estimator import QuantumKernelEstimator
    from .quantum.qsvc import QSVCWrapper

    model = QSVCWrapper(
        kernel=kernel, 
        use_hardware=use_hardware, 
        n_features=n_features, 
        random_state=random_state, 
        decision_function_shape=decision_function_shape,
        n_qubits=n_qubits
    )
    
    # kernel_instance = QuantumKernelEstimator(kernel_type=kernel, n_qubits=n_qubits, lambda_=lambda_, n_measurements=n_measurements)
    # feature_map = kernel_instance.build_quantum_kernel(n_features=n_features, use_hardware=use_hardware)
    # model = QSVC(quantum_kernel=feature_map, C=1.0, decision_function_shape='ovr')
    return model

def qesvc_model(kernel, random_state=42, use_hardware=False, n_features=20, n_qubits=4):
    from .quantum.qesvc import QESVC
    model = QESVC(kernel_type=kernel, use_hardware=use_hardware, n_features=n_features, random_state=random_state, n_qubits=n_qubits)
    return model

def libsvm_model(kernel, C=1.0, gamma=0.5, degree=3):
    from .libsvm.libsvm_model import LibSVMModel
    kernel_map = {'linear': 0, 'poly': 1, 'rbf': 2, 'sigmoid': 3}
    t_val = kernel_map[kernel]

    params = f"-t {t_val} -c {C} -b 1 -q"
    if kernel == 'linear':
        pass
    elif kernel == 'poly':
        params += f" -g {gamma} -d {degree}"
    elif kernel in ['rbf', 'sigmoid']:
        params += f" -g {gamma}"
    
    model = LibSVMModel(params)
    return model