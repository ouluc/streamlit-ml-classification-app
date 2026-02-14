def preprocess_data(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y