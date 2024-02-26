import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import xgboost as xgb

# Load data
df = pd.read_csv("loan_data.csv")

# Deal with null values
df.dropna(inplace=True)

# Dependent and independent data
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encoding categorical data
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# Splitting dataset to train_data and test_data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train[:, 12:] = sc.fit_transform(X_train[:, 12:])
X_test[:, 12:] = sc.transform(X_test[:, 12:])

# TRAIN 01: Logistic regression + Up-sampling with SMOTE
# Up-sampling with SMOTE
sm = SMOTE(k_neighbors=5)
X_train_sample, y_train_sample = sm.fit_resample(X_train, y_train)

# Train model
model_log = LogisticRegression()
model_log.fit(X_train_sample, y_train_sample)
y_pred_log = model_log.predict(X_test)

# Metrics and confusion matrix
print(classification_report(y_test, y_pred_log))
cm_log = confusion_matrix(y_test, y_pred_log, labels=model_log.classes_)
ConfusionMatrixDisplay(cm_log).plot()

# TRAIN 02: XGBoost + No Up-sampling
# Train model
model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=200)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# Metrics and confusion matrix
print(classification_report(y_test, y_pred_xgb))
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=model_xgb.classes_)
ConfusionMatrixDisplay(cm_xgb).plot()
