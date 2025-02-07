# %%
# Exploratory Data Analysis
import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv('data/Fraud.csv')

# %%
# Create data summary report and dashboard
# for data visualization
print(data.describe())
profile = ProfileReport(data, title='Transaction data')
profile.to_file('reports/transaction.html')


