import pandas as pd
import numpy as np

DATA_PATH = "data/taxi_outlier.csv"

df=pd.read_csv(DATA_PATH, quoting=3)

fare_amount = df['fare_amount']
passenger_count = df['passenger_count']
pickup_longitude = df['pickup_longitude']
pickup_latitude=df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']


def get_negative_index(list_data):
    neg_idx=[]
    
    for i, value in enumerate(list_data):
        if value < 0:
            neg_idx.append(i)
    return neg_idx

#Detecting customized outliers
def outlier_index():
    #if fare_amount and passenger_count is negative
    idx_fare_amount = get_negative_index(fare_amount)
    idx_passenger_count = get_negative_index(passenger_count)
    
    #if pickup and dropoff location is same
    idx_zero_distance=[]    
    idx= [i for i in range(len(passenger_count))]
    zipped = zip(idx, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude)
    
    for i, x, y, _x, _y in zipped:
        if (x,y) == (_x,_y):
            idx_zero_distance.append(i)
    
    #total indexes
    total_index4remove = list(set(idx_fare_amount+idx_passenger_count+idx_zero_distance))
    return total_index4remove


def remove_outlier(dataframe, list_idx):
    return dataframe.drop(idx for idx in list_idx)

df = df.drop(columns='Unnamed: 0')
df = df.drop(columns='Unnamed: 0.1')
df.info()

remove_index = outlier_index()
new=remove_outlier(df, remove_index)
new.to_csv("Data Analysis/Remove Outlier.csv")
new.info()
