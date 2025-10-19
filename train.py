import joblib
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
print(classification_report(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, clf.predict_proba(X_test_s)[:,1]))
joblib.dump({'model':clf, 'scaler':scaler}, 'models/disease_rf.joblib')
print('Saved model to models/disease_rf.joblib')



