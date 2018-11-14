import csv
import pandas as pd
df = pd.read_csv("all_cities_combined2.csv")
keep_col = ['O3_AQI', 'id', 'State Code']
df = df[keep_col]
df = df.loc[df.groupby('id')['O3_AQI'].idxmax()]

df.to_csv("max_by_county_O3.csv", index=False)

