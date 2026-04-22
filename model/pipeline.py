import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from .models import quantum_model, svc_model, xgboost_model, catboost_model, libsvm_model
from .training import train_model, validate_model, test_model
from utils.hpo import build_hpo

def run_model(data, kernels, model_type='quantum', use_hardware=False, stage='cv', use_hpo=False, random_state=42, n_splits=None, fold=None):
    results = []
    
    if model_type in ['quantum', 'sklearn', 'libsvm']: kernel_list = kernels
    else: kernel_list = [None]
    for kernel in kernel_list:
        if model_type == 'quantum':
            model = quantum_model(kernel, n_measurements=1024, use_hardware=use_hardware, n_features=data['X_train'].shape[1])
        elif model_type == 'sklearn':
            model = svc_model(kernel, random_state=random_state)
        elif model_type == 'xgboost':
            model = xgboost_model(n_estimators=500, max_depth=10, subsample=0.8, learning_rate=1, random_state=random_state)
        elif model_type == 'catboost':
            model = catboost_model(iterations=1000, learning_rate=0.1, depth=6, random_seed=random_state)
        elif model_type == 'libsvm':
            model = libsvm_model(kernel, C=1.0, gamma=0.5, degree=3)

        # HPO/TRAIN
        best_params = None
        best_model = None
        train_start = time.time()
        if use_hpo and model_type in ['quantum', 'sklearn', 'xgboost', 'catboost']:
            opt = build_hpo(model_type, model, kernel=kernel, random_state=random_state, n_splits=n_splits)
            opt.fit(data['X_train'], data['y_train'])
            best_model = opt.best_estimator_
            best_params = opt.best_params_

            print(f"\nFold {fold} | Kernel={kernel} | Best Params: {best_params}")
        else:
            model = train_model(model, model_type, data)
        train_time = time.time() - train_start

        # # VAL
        # val_start = time.time()
        # val_acc = validate_model(model, data)
        # val_time = time.time() - val_start

        # TEST
        test_start = time.time()
        if use_hpo and model_type in ['quantum', 'sklearn', 'xgboost', 'catboost']:
            y_test_pred = test_model(best_model, data)
        else:
            y_test_pred = test_model(model, data)
        test_time = time.time() - test_start
            
        total_time = train_time + test_time

        cm = confusion_matrix(data['y_test'], y_test_pred)

        report = classification_report(data['y_test'], y_test_pred, output_dict=True)
        results.append({
            'kernel': kernel,
            # 'val_accuracy': val_acc,
            'accuracy': report['accuracy'],
            'weighted_precision': report['weighted avg']['precision'],
            'weighted_recall': report['weighted avg']['recall'],
            'weighted_f1-score': report['weighted avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1-score': report['macro avg']['f1-score'],
            'train_time': train_time,
            # 'val_time': val_time,
            'test_time': test_time,
            'total_time': total_time,
            'confusion_matrix': cm,
            'best_params': best_params,
            'best_model': best_model,
        })

    return results