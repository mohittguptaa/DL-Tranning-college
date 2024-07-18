import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
# from tensorflow.keras.optimizers import SGO
from tensorflow.random import set_seed
set_seed(455)
np.random.seed(455)

data=pd.read_csv("Mastercard_stock_history.csv",index_col="Date",
                 parse_dates=["Date"])
datas=data.drop(['Dividends','Stock Splits'],axis=1)
# print(datas)
# print(datas.describe())
# print(datas.isna().sum())

tstart,tend=2016,2020
def train_test_plot(datas,tstart,tend):
    datas.loc[f'{tstart}':f'{tend}',"High"].plot(legend=True)
    datas.loc[f'{tend+1}':,"High"].plot(legend=True)
    plt.legend([f"Train (Before {tend+1})",f"Test ({tend+1} and beyond)"])
    plt.show()

train_test_plot(datas,tstart,tend)

# Data Preprocessing

def train_test_split(datas,tstart,tend):
    train=datas.loc[f'{tstart}':f'{tend}',"High"].values
    test=datas.loc[f'{tend+1}':,"High"]
    return train,test

train,test=train_test_split(datas,tstart,tend)
print("Train Data::",train)