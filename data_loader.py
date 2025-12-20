import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_dataset(filepath):
    if "monks" in filepath:
        if "train" in filepath:
            test_path = filepath.replace("train", "test")
            train_df = pd.read_csv(filepath, sep=" ", header=None)
            test_df = pd.read_csv(test_path, sep=" ", header=None)
            
            train_df = train_df.dropna(axis=1)
            test_df = test_df.dropna(axis=1)
            
            # MONK’s format:
            # col 0  -> class
            # col 1–6 -> attributes
            # col 7  -> id (drop)

            X_train_raw = train_df.iloc[:, 1:7].values.astype(np.int32)
            y_train     = train_df.iloc[:, 0].values.astype(np.int32)

            X_test_raw  = test_df.iloc[:, 1:7].values.astype(np.int32)
            y_test      = test_df.iloc[:, 0].values.astype(np.int32)

            
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            X_train = encoder.fit_transform(X_train_raw)
            X_test = encoder.transform(X_test_raw)
            
            return X_train, X_test, y_train, y_test
        else:
            return None
    
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(np.int32)
    
    if "he_exact2_fail" in filepath:
        scaler = StandardScaler()
        X = scaler.fit_transform(X.astype(np.float64))
        X = X.astype(np.float64)
    else:
        X = X.astype(np.float64)
    
    if len(X) < 100:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test
