NEW = False
def preprocess(data, fit_scaler=True, scaler=None, target_col='Kategori'):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.decomposition import PCA

    if NEW:
        cpdata = data.copy().reset_index(drop=True) 

        cat_feature_cols = [col for col in cpdata.columns if pd.api.types.is_string_dtype(cpdata[col]) and col != target_col]
        num_feature_cols = [col for col in cpdata.columns if not pd.api.types.is_string_dtype(cpdata[col]) and col != target_col]
        
        label_encoder = LabelEncoder()
        for col in cat_feature_cols:
            cpdata[col] = label_encoder.fit_transform(cpdata[col])
        
        cpdata[target_col] = label_encoder.fit_transform(cpdata[target_col])

        if fit_scaler:
            scaler = StandardScaler()
            cpdata[num_feature_cols] = scaler.fit_transform(cpdata[num_feature_cols])
        else:
            cpdata[num_feature_cols] = scaler.transform(cpdata[num_feature_cols])

        return cpdata, scaler
    else:
        # reset index
        X_train = data['X_train'].reset_index(drop=True)
        # X_val  = data['X_val'].reset_index(drop=True)
        X_test  = data['X_test'].reset_index(drop=True)
        y_train = data['y_train'].reset_index(drop=True)
        # y_val = data['y_val'].reset_index(drop=True)
        y_test  = data['y_test'].reset_index(drop=True)

        cat_features = [
            i for i, col in enumerate(X_train.columns)
            if X_train[col].dtype == 'object'
        ]

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        # y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)

        # z score normalization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        # X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # PCA untuk reduksi dimensi (opsional, bisa di-tuning)
        pca = PCA(n_components=5)
        X_train = pca.fit_transform(X_train)
        # X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

        data_processed = {
            'X_train': X_train,
            # 'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            # 'y_val': y_val,
            'y_test': y_test,
            'cat_features': cat_features,
            'pca': pca,
            'classes': label_encoder.classes_
        }

        return data_processed
    

def encode_labels(data):
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    cat_feature_cols = [col for col in data.columns if pd.api.types.is_string_dtype(data[col])]

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(cat_feature_cols)
    return encoded

def scale_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled