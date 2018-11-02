#code to take all generated files from main2.py and combine into a single csv

import glob

#get list of csv's in /city_data
allfiles = (glob.glob("../city_data/*.csv"))

fout=open("new_all_cities_combined.csv","a")

# first file, keep header
for line in open(allfiles[0]):
    fout.write(line)

# combine rest of files
for s in range(1, len(allfiles), 1):
    f = open(allfiles[s])
    f.readline() # skip the header
    for line in f:
         fout.write(line)
    f.close()
fout.close()