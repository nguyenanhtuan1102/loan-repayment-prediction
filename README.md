# Loan Data Classification with Logistic Regression and XGBoost

![epay-your-loans-on-time](https://github.com/tuanng1102/loan-repayment-prediction/assets/147653892/823dc2e6-426c-40b1-b7aa-914dd954a971)

This code explores two machine learning models, Logistic Regression and XGBoost, for classifying loan applications based on loan data.

## Import Libraries

``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import xgboost as xgb
```

## Data Loading and Preprocessing:

### 1. Data Loading: 
The code loads the loan data from "loan_data.csv" using Pandas.

``` bash
df = pd.read_csv("loan_data.csv")
```

### 2. Missing Value Handling: 
Missing values are removed using dropna.

``` bash
df.dropna(inplace=True)
```

### 3. Feature Separation: 
Features (independent variables) and target variable (dependent variable) are separated:

- X: Features
- y: Target variable (loan approval status)

``` bash
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
```

### 4. Encoding Categorical Data: 
The categorical feature in column 1 is encoded using OneHotEncoder.

``` bash
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))
```

### 5. Data Splitting: 
The data is split into training and testing sets for model evaluation using train_test_split.

``` bash
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 6. Feature Scaling (Numerical Features): 
Numerical features (except the first 12) are standardized using StandardScaler.

``` bash
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,12:] = sc.fit_transform(X_train[:,12:])
X_test[:,12:] = sc.transform(X_test[:,12:])
```

## Model Training and Evaluation:

### 1. Logistic Regression with SMOTE:

- SMOTE (Synthetic Minority Oversampling Technique) is applied to address potential class imbalance in the training data.
- An SMOTE instance with k_neighbors=5 is created to generate synthetic samples for the minority class.
- The up-sampled training data (X_train_sample, y_train_sample) is used to train a Logistic Regression model.
- The model's performance is evaluated using classification report and confusion matrix.

``` bash
# Upsampling with SMOTE
sm = SMOTE(k_neighbors=5)
X_train_sample, y_train_sample = sm.fit_resample(X_train,y_train)

# Train model
model_log = LogisticRegression()
model_log.fit(X_train_sample, y_train_sample)
y_pred_log = model_log.predict(X_test)

# Metrics and confusion matrix
print(classification_report(y_test, y_pred_log))
cm_log = confusion_matrix(y_test, y_pred_log, labels=model_log.classes_)
ConfusionMatrixDisplay(cm_log).plot()
```

![16](https://github.com/tuanng1102/loan-repayment-prediction/assets/147653892/cdd056fe-53c0-492f-a505-36f609d47ef7)

![14](https://github.com/tuanng1102/loan-repayment-prediction/assets/147653892/31255208-24ee-4745-b28d-4236fc0afbbb)

### 2. XGBoost with NO SMOTE:

- An XGBoost model with n_estimators=200 and random state set to 42 is trained on the original training data (X_train, y_train).
- The model's performance is evaluated using classification report and confusion matrix.

``` bash
# Train model
model_xgb = xgb.XGBClassifier(random_state=42, n_estimators=200)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_test)

# Metrics and confusion matrix
print(classification_report(y_test, y_pred_xgb))
cm_xgb = confusion_matrix(y_test, y_pred_xgb, labels=model_xgb.classes_)
ConfusionMatrixDisplay(cm_xgb).plot()
```

![17](https://github.com/tuanng1102/loan-repayment-prediction/assets/147653892/3dce8d76-8a44-4c95-a212-42c89b6a7798)

![15](https://github.com/tuanng1102/loan-repayment-prediction/assets/147653892/09361c3a-a0dc-4ea1-8223-0d0092d50a4e)

The confusion matrix visualizations are not included in the code snippet, but they are generated using ConfusionMatrixDisplay.plot().
Overall, this code demonstrates the application of two different machine learning models for loan data classification. Comparing the models' performance based on classification reports and confusion matrices can help you choose the most suitable model for your specific problem.

