from skopt.space import Real, Integer
from skopt import BayesSearchCV
def get_search_space(model_type, kernel=None, booster=None):
  if model_type == 'sklearn':
    base = {
      'svc__C': Real(1e-3, 1e3, prior='log-uniform'),
      'svc__class_weight': ['balanced', None],
      'svc__decision_function_shape': ['ovr', 'ovo'],
      'svc__shrinking': [True, False],
      'svc__tol': Real(1e-5, 1, prior='log-uniform')
    }

    gamma = Real(1e-4, 1e-1, prior='log-uniform')
    coef0 = Real(0, 1)

    if kernel == 'linear':
      return {
        **base
      }
    elif kernel == 'poly':
      return {
        **base,
        'svc__degree': Integer(2, 5),
        'svc__gamma': gamma,
        'svc__coef0': coef0,
      }
    elif kernel == 'rbf':
      return {
        **base,
        'svc__gamma': gamma,
      }
    elif kernel == 'sigmoid':
      return {
        **base,
        'svc__gamma': gamma,
        'svc__coef0': coef0,
      }
  
  elif model_type == 'xgboost':
    base = {
        'xgb__n_estimators':  Integer(200, 2000),
        'xgb__learning_rate': Real(1e-3, 0.5, prior='log-uniform'),
    }

    max_depth = Integer(3, 20)
    subsample = Real(0.0, 1.0)
    grow_policy = ['depthwise', 'lossguide']
    gamma = Real(1e-4, 1, prior='log-uniform')
    colsample_bytree = Real(0.0, 1.0)

    if booster == 'gbtree':
      return {
        **base,
        'xgb__max_depth': max_depth,
        'xgb__subsample': subsample,
        'xgb__grow_policy': grow_policy,
        'xgb__tree_method': ['exact', 'approx', 'hist'],
        'xgb__gamma': gamma,
        'xgb__colsample_bytree': colsample_bytree,
    }
    elif booster == 'dart':
      return {
        **base,
        'xgb__max_depth': max_depth,
        'xgb__subsample': subsample,
        'xgb__grow_policy': grow_policy,
        'xgb__gamma': gamma,
        'xgb__colsample_bytree': colsample_bytree,
        # dart-specific
        'xgb__rate_drop': Real(1e-4, 0.3, prior='log-uniform'),
        'xgb__skip_drop': Real(0.0, 0.5),
        'xgb__sample_type': ['uniform', 'weighted'],
        'xgb__normalize_type': ['tree', 'forest'],
    }

  elif model_type == 'catboost':
    pass
  elif model_type == 'quantum':
    return {
      'qsvc__C': Real(1e-3, 1e3, prior='log-uniform'),
      'qsvc__class_weight': ['balanced', None],
      'qsvc__decision_function_shape': ['ovr', 'ovo'],
      'qsvc__n_qubits': [5, 8, 10],
      'qsvc__lambda_': Real(1e-3, 1e3, prior='log-uniform'),
      'qsvc__n_measurement': [256, 1024, 2048],
    }

def build_hpo(model_type, cv, model, kernel=None, random_state=42):
  search_space = get_search_space(model_type, kernel)
  opt = BayesSearchCV(
    model,
    search_spaces=search_space,
    n_iter=32,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=random_state,
  )
  return opt