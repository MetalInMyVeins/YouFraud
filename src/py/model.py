# Before data preprocessing, let's talk about features of fraud
# transactions.
# 
# There is no point in transferring money from account having
# ZERO balance. Fraud.
#
# Transactions where large amount of money are entirely cashed
# out or transferred are fraud transactions.
#
# Transactions which are flagged fraud are fraud transactions.

# Analyzing EDA Report:
# > No missing values
# > No duplicate rows
# > High correlation in some fields
# > isFraud is highly imbalanced
# > amount is highly skewed
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset in DataFrame
data = pd.read_csv('data/Fraud.csv')
df = pd.DataFrame(data)

# %%
# Check columns
print(df.columns)
# Check data types
print(df.dtypes)

# %%
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].corr())
# Following fields show high correlation
# (newBalanceOrig, oldBalanceOrg) = 0.998803
# (newbalanceDest, oldbalanceDest) = 0.976569

# %%
print(len(df[df['amount'] == 0]))
# 16 transactions have amount == 0. These are fraud transactions
# because it is not normal to transfer money from an empty
# account. This means fraud parties are brute forcing accounts
# without even knowing how much balance they contain.

# %%
print(len(df[(df['amount'] == df['oldbalanceOrg']) & (df['isFraud'] == 1)]))
# 8034 transactions are fraud where entire money is transferred or
# cashed out from account.

# %%
# count number of isFraud
print(df['isFraud'].value_counts())
# isFraud
# 0    6354407
# 1       8213
# Name: count, dtype: int64
# Shows that isFraud is highly imbalanced.

# %%
# generate a single boxplot for outlier detection of all columns
plt.figure(figsize=(15, 10))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.savefig('reports/outliers.png')

# %%
# calculate inter quartile range for 'step' column
q1 = df['step'].quantile(0.25)
q3 = df['step'].quantile(0.75)
iqr = q3 - q1
# calculate upper and lower bounds
upper_bound = q3 + 1.5 * iqr
lower_bound = q1 - 1.5 * iqr
# count number of outliers
print(len(df[(df['step'] < lower_bound) | (df['step'] > upper_bound)]))
# Significant number of outliers in 'step' column. But removing
# them will result in loss of data. So, we will keep them.

# %%
# generate heatmap for correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.savefig('reports/correlation.png')

# %%
# Feature Selection
# remove highly correlated columns
df.drop(['newbalanceOrig', 'newbalanceDest'], axis=1, inplace=True)
print(df.columns)

# %%
# calculate number of unique values in categorical columns
cat_cols = df.select_dtypes('object').columns
print(df[cat_cols].nunique())
# dropping irrelavant categorical columns
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# %%
# after dropping, there is one categorical column left.
# rate them by their unique values
print(df['type'].value_counts())
# replace large counts with bigger numeric numbers
df['type'] = df['type'].replace(['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], [4, 3, 2, 1, 0])

# %%
# handle unbalanced data
normal_transaction = df[df['isFraud'] == 0]
fraud_transaction = df[df['isFraud'] == 1]

# randomly select 8213 normal transactions
normal_transaction = normal_transaction.sample(n=fraud_transaction.shape[0])
print(pd.DataFrame(normal_transaction['amount']).describe())
print(pd.DataFrame(fraud_transaction['amount']).describe())

# %%
# combine normal and fraud transactions
new_df = pd.concat([normal_transaction, fraud_transaction])
# shuffle the dataset
new_df = new_df.sample(frac=1, random_state=42)
# save the new dataset
new_df.to_csv('data/FraudProcessed.csv', index=False)

# %%
# load the new dataset
fraud_data = pd.read_csv('data/FraudProcessed.csv')
fraud_df = pd.DataFrame(fraud_data)

# train test split
from sklearn.model_selection import train_test_split
x = fraud_df.drop('isFraud', axis=1)
y = fraud_df['isFraud']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# isFraud values are evenly distributed in train and test datasets

# %%
# feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_test_scaler = scaler.transform(x_test)

# %%
# Train and evaluate models
# train and test logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_reg = LogisticRegression()
log_reg.fit(x_train_scaler, y_train)
y_pred = log_reg.predict(x_test_scaler)
print(f'Logistic regression accuracy: {accuracy_score(y_test, y_pred)}')
# 0.80

# train and test random forest model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train_scaler, y_train)
y_pred = rf.predict(x_test_scaler)
print(f'Random forest accuracy: {accuracy_score(y_test, y_pred)}')
# 0.98

# train and test svm model
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train_scaler, y_train)
y_pred = svm.predict(x_test_scaler)
print(f'SVM accuracy: {accuracy_score(y_test, y_pred)}')
# 0.90
# So we're selecting random forest model.

# %%
# save model
import joblib

joblib.dump(rf, 'models/fraud_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')


