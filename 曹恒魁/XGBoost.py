# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation,metrics
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import f_regression
from xgboost.sklearn import XGBRegressor
	
if __name__ == '__main__':
    data = pd.read_csv('train_data.csv',usecols=['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1'],dtype=np.float64)
    result = data['MidPrice'].copy()
    data['MidPrice'] = data['MidPrice'].shift(-1)-data['MidPrice']
    newToVolume = data['Volume'].shift(-1)-data['Volume']
    data.fillna(0,inplace=True)
    NewestPrice = data['BidPrice1'] - data['AskPrice1']
    NewestVolume= data['BidVolume1'] - data['AskVolume1']
    data.insert(0,'NewestPrice',NewestPrice)
    data.insert(0,'NewestVolume',NewestVolume)
    data.insert(0,'newToVolume',newToVolume)
    data.fillna(0,inplace=True)
    predictors = [x for x in data.columns if x not in ['MidPrice']]
    for i in predictors:
        data[i] = preprocessing.scale(data[i])
    train=data
    train_result=result
	
    data = pd.read_csv('test_data.csv',usecols=['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1'],dtype=np.float64)
    result = data['MidPrice'].copy()
    data['MidPrice'] = data['MidPrice'].shift(-1)-data['MidPrice']
    newToVolume = data['Volume'].shift(-1)-data['Volume']
    data.fillna(0,inplace=True)
    NewestPrice = data['BidPrice1'] - data['AskPrice1']
    NewestVolume= data['BidVolume1'] - data['AskVolume1']
    data.insert(0,'NewestPrice',NewestPrice)
    data.insert(0,'NewestVolume',NewestVolume)
    data.insert(0,'newToVolume',newToVolume)
    data.fillna(0,inplace=True)
    predictors = [x for x in data.columns if x not in ['MidPrice']]
    for i in predictors:
        data[i] = preprocessing.scale(data[i])
    test=data
    test_result=result

    predictors = [x for x in train.columns if x not in ['MidPrice']]
    model = XGBRegressor(learning_rate =0.02,n_estimators=1000,max_depth=5,objective= 'reg:linear',silent=True)
    model.fit(train[predictors].values,train['MidPrice'].values)
    ans = model.predict(test[predictors].values)
    f = open('predict.csv',mode= 'w')
    f.write('caseid,midprice')
    f.write('\n')
    i = 142
    while i*10+9<10000:
        temp=ans[i*10+9]+test_result[i*10+9]
        f.write(str(i+1)+','+str(temp))
        f.write('\n')
        i += 1
    print(mean_squared_error(test['MidPrice'].values,ans))
    f.close()
