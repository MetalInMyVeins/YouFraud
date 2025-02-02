import pandas as pd

data = pd.read_csv('data/ExampleFraud.csv')
df = pd.DataFrame(data)
print(df.head())


