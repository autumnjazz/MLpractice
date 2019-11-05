import pandas as pd
import numpy as np

DATA_PATH = 'data/taxi_test.csv'

df= pd.read_csv(DATA_PATH)
print("\n#Before removing missing data")
df.info()

def remove_NaN(df_data):
    index_remove=[]
    for key in df_data.keys():
        for i, data in enumerate(df_data[key]):
            if pd.isna(data) or pd.isnull(data):
                index_remove.append(i)
                
    index_remove = list(set(index_remove))
    
    removed_data = df_data.drop(idx for idx in index_remove)
    return removed_data
    

removed_df= remove_NaN(df)
print("\n#After removing missing data")
removed_df.info()

 