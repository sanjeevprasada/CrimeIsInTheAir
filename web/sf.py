import pandas as pd

df = pd.read_csv('state_CO.csv')
keep_col = df[['State_Code','CO_AQI']]
df2 = keep_col.sort_values(by = ['CO_AQI'],ascending=False)
print df2