import pandas as pd
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import time
import datetime as dt
from numpy import arange
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# data engineering
AMZN = pd.read_csv('AAPL.csv')
#AMZN['name'] = 'AMZN'
rename_dict = {
    'High': 'AMZN_high',
    'Low': 'AMZN_low',
    'Close': 'AMZN_close',
    'Open': 'AMZN_open',
    'Adj Close': 'AMZN_adj_close',
    'Volume': 'AMZN_volume'
}
AMZN = AMZN.rename(columns=rename_dict)

AAPL = pd.read_csv('AMZN.csv')
#AAPL['name'] = 'AAPL'
rename_dict = {
    'High': 'AAPL_high',
    'Low': 'AAPL_low',
    'Close': 'AAPL_close',
    'Open': 'AAPL_open',
    'Adj Close': 'AAPL_adj_close',
    'Volume': 'AAPL_volume'
}
AAPL = AAPL.rename(columns=rename_dict)

GOOG = pd.read_csv('GOOG.csv')
#GOOG['name'] = 'GOOG'
rename_dict = {
    'High': 'GOOG_high',
    'Low': 'GOOG_low',
    'Close': 'GOOG_close',
    'Open': 'GOOG_open',
    'Adj Close': 'GOOG_adj_close',
    'Volume': 'GOOG_volume'
}
GOOG = GOOG.rename(columns=rename_dict)

TSLA = pd.read_csv('TSLA(1).csv')
#TSLA['name'] = 'TSLA'
rename_dict = {
    'High': 'TSLA_high',
    'Low': 'TSLA_low',
    'Close': 'TSLA_close',
    'Open': 'TSLA_open',
    'Adj Close': 'TSLA_adj_close',
    'Volume': 'TSLA_volume'
}
TSLA = TSLA.rename(columns=rename_dict)
merged1 = pd.merge(AMZN, AAPL, on=['Date'], how='outer')
merged2 = pd.merge(GOOG, TSLA, on=['Date'], how='outer')

concat1 = pd.concat([AMZN[['AMZN_high', 'AMZN_low', 'AMZN_close', 'AMZN_open', 'AMZN_adj_close', 'AMZN_volume']], AAPL[['AAPL_high', 'AAPL_low', 'AAPL_close', 'AAPL_open', 'AAPL_adj_close', 'AAPL_volume']]], axis=1)
concat2 = pd.concat([GOOG[['GOOG_high', 'GOOG_low', 'GOOG_close', 'GOOG_open', 'GOOG_adj_close', 'GOOG_volume']], TSLA[['TSLA_high', 'TSLA_low', 'TSLA_close', 'TSLA_open', 'TSLA_adj_close', 'TSLA_volume']]], axis=1)

data = pd.merge(merged1, merged2, on=['Date'], how='outer')
data = pd.concat([data, concat1, concat2], axis=1)

data = data.dropna()
data['date'] = pd.to_datetime(data['Date'])
data['date_int'] = (data['date'] - data['date'].min()).dt.days
data['day_of_week'] = data['date'].dt.day_name()
# monday = 0, tuesday = 1
data['day_of_week_int'] = data['date'].dt.dayofweek
data['month_int'] = data['date'].dt.month
data['month_str'] = data['date'].dt.strftime('%B')
seasons = [(1, 'winter'), (4, 'spring'), (7, 'summer'), (10, 'fall')]
season_dict = dict(zip(range(1, 13), [s for m, s in seasons for s in [s]*3] ))
data['season_int'] = data['date'].dt.month.apply(lambda x: (x-1)//3 + 1)
data['season_str'] = data['date'].dt.month.map(season_dict)

#grouped = data.groupby('date')
print(data.head(5))
data.to_csv('complete_data.csv', index=False)
