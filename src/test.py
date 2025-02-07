# %%
# test fraud model
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# load model
fraud_model = joblib.load('models/fraud_model.pkl')
# load scaler file to scale the input array
scaler = joblib.load('models/scaler.pkl')

# create a sample input array
input_data = {'step': [1], 'type': [4], 'amount': [181087.0], 'oldbalanceOrg': [181.0], 'oldbalanceDest': [0.0], 'isFlaggedFraud': [0]}
input_df = pd.DataFrame(input_data)

# scale the input array
input_df = scaler.transform(input_df)

# predict the output
isFraud = fraud_model.predict(input_df)
print(f'YouFraud: {isFraud[0]}')
# YouFraud: 1
# Means transaction is fraud.
# YouFraud: 0
# Means transaction is not fraud.


