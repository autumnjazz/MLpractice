import numpy as np
import pandas as pd

df=pd.read_csv("data/taxi_outlier.csv", quoting=3)

# ['2009-06-15 17:26:21 UTC', ...]
pickup_datetime = df['pickup_datetime'] 
year_date=[]
time=[]

# '2009-06-15' '17:26:21'
mod1 = [pickup_datetime[i].split(' ') for i in range(len(pickup_datetime))]
for i in range(len(mod1)):
    year_date.append(mod1[i][0])
    time.append(mod1[i][1])

# '2009' '06' '15'
lgt = len(year_date)
years=[year_date[i].split('-')[0] for i in range(lgt)]
months=[year_date[i].split('-')[1] for i in range(lgt)]
days=[year_date[i].split('-')[2] for i in range(lgt)]

#'17'
lgt = len(time)
hours=[time[i].split(':')[0] for i in range(lgt)]

print(years[:10])
print(months[:10])
print(days[:10])
print(hours[:10])