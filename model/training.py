def train_model(model, model_type, data):
  if model_type == 'xgboost':
      model.fit(
          data['X_train'], 
          data['y_train'], 
          eval_set=[(data['X_test'], data['y_test'])], 
          verbose=False
      )
  elif model_type == 'catboost':
      model.fit(
          data['X_train'],
          data['y_train'],
          cat_features=data['cat_features'],
          eval_set=(data['X_test'], data['y_test']),
          early_stopping_rounds=50
      )
  else:
      model.fit(data['X_train'], data['y_train'])
  return model

def validate_model(model, data):
  from sklearn.metrics import accuracy_score
  y_val_pred = model.predict(data['X_val'])
  val_acc = accuracy_score(data['y_val'], y_val_pred)
  return val_acc

def test_model(model, data):
  y_test_pred = model.predict(data['X_test'])
  return y_test_pred