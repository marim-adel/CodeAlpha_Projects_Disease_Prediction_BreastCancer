import joblib, numpy as np, pandas as pd
def load_model(path='models/disease_rf.joblib'):
    data = joblib.load(path)
    return data['model'], data['scaler']
def predict(model, scaler, X_df):
    # X_df: pandas DataFrame or 2D-array with same feature order as training (breast cancer dataset)
    import pandas as pd, numpy as np
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    Xs = scaler.transform(X_df)
    prob = model.predict_proba(Xs)[:,1]
    pred = model.predict(Xs)
    return pred, prob
