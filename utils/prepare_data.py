from .load_dataset import load_dataset
from .preprocess import preprocess
def prepare_data(feature_cols, target_cols, dataset_path=None, random_state=None, manual=False, manualDirPath=None):
  X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(
    feature_cols, 
    target_cols, 
    manual=manual, 
    path=dataset_path,
    random_state=random_state,
    manualDirPath=manualDirPath
  )
  data = {
      'X_train': X_train,
      'X_val': X_val,
      'X_test': X_test,
      'y_train': y_train,
      'y_val': y_val,
      'y_test': y_test
  }

  data_processed = preprocess(data)

  return data_processed