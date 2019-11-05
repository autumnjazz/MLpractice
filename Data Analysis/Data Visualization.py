import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")

df = pd.read_csv('Data Analysis/Remove Outlier.csv', quoting=3)
pickup_longitude = df['pickup_longitude']
pickup_latitude = df['pickup_latitude']
dropoff_longitude = df['dropoff_longitude']
dropoff_latitude = df['dropoff_latitude']

# calculate distance between pickup and dropoff
def distance(pick_lat, pick_lon, drop_lat, drop_lon):
    p = 0.0174 ## Pi/180
    ## Haversine Formula
    a = 0.5 - np.cos((drop_lat - pick_lat) * p)/2 + np.cos(pick_lat * p) * np.cos(drop_lat * p) * (1 - np.cos((drop_lon - pick_lon) * p)) / 2
    return 0.621 * 12742 * np.arcsin(np.sqrt(a))

list_distance = distance(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)
df['distance'] = list_distance

plt.figure(figsize=(16,12))
axs = plt.subplot()

#graph showing fare amount by distance
axs.scatter(df.distance, df.fare_amount, alpha=0.2)
axs.set_xlabel('distance')
axs.set_ylim(0,)
axs.set_ylabel('fare $USD')
axs.set_title('All data')

plt.savefig("Data Analysis/Data Visualization.png")
