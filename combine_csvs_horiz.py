"""Code to combine all csv's from /web named by county id into csvs for each pollutant """

import csv
import pandas as pd
import numpy as np

dfdates = pd.read_csv("dates.csv")
dfdates = dfdates.drop_duplicates()

dates = dfdates['Date Local'].tolist()

c = [4013, 4019, 6013, 6025, 6029, 6037, 6059, 6065, 6067, 6071, 6073, 6075, 6083, 6087, 6095, 6111,
    6001, 6019, 6023, 6085, 8001, 8031, 8057, 11001, 12095, 12057, 17031, 17163, 18097, 20107, 20191, 20209,
    21019, 21059, 21067, 21101, 21111, 21145, 22033, 22051, 26163, 26081, 29189, 29510, 34007, 34013, 36005,
    36103, 36101, 36055, 37067, 37119, 37183, 40021, 40071, 40115, 40143, 40001, 40109, 42003, 42007, 42011,
    42013, 42017, 42021, 42069, 42071, 42073, 42091, 42095, 42101, 42125, 42129, 42133, 42049, 42043, 42079,
    42001, 48113, 48141, 48201, 48309, 48029, 48453, 51059, 51510, 51087, 51650, 51161, 25025, 32003, 32031,
    33011, 33015, 47075, 47121, 47009, 45019, 45079, 9009, 9001, 9003, 9005, 23003, 24005, 24033, 24023, 55079,
    80002, 5119, 19153, 41051, 56041, 56021, 56037, 56013, 38017, 16001, 39009, 39103, 39035, 39061, 13089,
    10003, 15003, 27003, 35001, 44007, 46099, 46127, 49035, 49013, 49047, 1073, 53033, 2090]

ids = pd.DataFrame({'id':c})
dcol = pd.DataFrame(np.nan, index=np.arange(0,len(c)), columns=dates)
df = pd.concat([ids,dcol], axis=1)

for f in range(0,len(c)-1):
    fname = str(c[f])+'.csv'
    df1 = pd.read_csv(fname)
    for i in range(0,df1.shape[0]-1):
        for n in range(0,len(dates)-1):
            if (df1.iat[i,0]  == dates[n]):
                df.iat[f, n+1] = df1.iat[i, 1]
                #print(df1.iat[i, 0])
print(df)
df.to_csv("no2_all.csv", index=False)

for f in range(0,len(c)-1):
    fname = str(c[f])+'.csv'
    df1 = pd.read_csv(fname)
    for i in range(0,df1.shape[0]-1):
        for n in range(0,len(dates)-1):
            if (df1.iat[i,0]  == dates[n]):
                df.iat[f, n+1] = df1.iat[i, 2]
                #print(df1.iat[i, 0])
print(df)
df.to_csv("o3_all.csv", index=False)

for f in range(0,len(c)-1):
    fname = str(c[f])+'.csv'
    df1 = pd.read_csv(fname)
    for i in range(0,df1.shape[0]-1):
        for n in range(0,len(dates)-1):
            if (df1.iat[i,0]  == dates[n]):
                df.iat[f, n+1] = df1.iat[i, 3]
                #print(df1.iat[i, 0])
print(df)
df.to_csv("so2_all.csv", index=False)

for f in range(0,len(c)-1):
    fname = str(c[f])+'.csv'
    df1 = pd.read_csv(fname)
    for i in range(0,df1.shape[0]-1):
        for n in range(0,len(dates)-1):
            if (df1.iat[i,0]  == dates[n]):
                df.iat[f, n+1] = df1.iat[i, 4]
                #print(df1.iat[i, 0])
print(df)
df.to_csv("co_all.csv", index=False)
