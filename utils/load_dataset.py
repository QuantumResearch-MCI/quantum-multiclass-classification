from sklearn.model_selection import train_test_split
import pandas as pd



def load_dataset(feature_cols, target_cols, manual=False, path=None, manualDirPath=None, random_state=None):

    if manual:
        train_data = pd.read_csv(f'{manualDirPath}/train.csv')
        val_data = pd.read_csv(f'{manualDirPath}/val.csv')
        test_data = pd.read_csv(f'{manualDirPath}/test.csv')
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_cols]

        X_val = val_data[feature_cols]
        y_val = val_data[target_cols]

        X_test = test_data[feature_cols]
        y_test = test_data[target_cols]
    else:
        print("Stratify")
        df = pd.read_csv(path)
        
        X = df[feature_cols]
        y = df[target_cols]

        # random_state = [42, x, x, x,]
        # train:temp (70%:30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=random_state
        )

        # temp -> val:test = 15:15
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )

    return X_train, X_val, X_test, y_train, y_val, y_test