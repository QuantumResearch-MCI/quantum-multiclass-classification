from .pipeline import run_model

def evaluate_svc(data, kernel_types, use_hpo=False, n_splits=5, fold=None):
    result = run_model(
        data, 
        kernel_types, 
        model_type='sklearn', 
        use_hpo=use_hpo,
        n_splits=n_splits,
        fold=fold,
    )
    return result

def evaluate_xgboost(data, kernel_types):
    result = run_model(
        data, 
        kernel_types, 
        model_type='xgboost', 
    )
    return result

def evaluate_catboost(data, kernel_types):
    result = run_model(
        data, 
        kernel_types, 
        model_type='catboost', 
    )
    return result

def evaluate_quantum(data, quantum_kernel_types, use_hardware=False):
    result = run_model(
        data, 
        kernels=quantum_kernel_types, 
        model_type='quantum', 
        use_hardware=use_hardware,
    )
    return result

def evaluate_libsvm(data, kernel_types):
    result = run_model(
        data, 
        kernels=kernel_types, 
        model_type='libsvm', 
    )
    return result