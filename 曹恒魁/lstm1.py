# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras import optimizers
from sklearn import preprocessing

			
if __name__ == '__main__':
    sc = preprocessing.MinMaxScaler(feature_range = (-0.5,0.5))
    sc1 = preprocessing.MinMaxScaler(feature_range = (-0.5,0.5))
    file = pd.read_csv('train_data.csv')
    data = file[['MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1']]
    labels = file[['MidPrice']]
    data = np.array(data)
    data = sc.fit_transform(data)
    labels = np.array(labels)
    labels = sc1.fit_transform(labels)
    features=data
    L = len(labels) - 30
    f = list()
    l = list()
    for i in range(L):
        feature_for_case = features[i: i + 10]
        label_for_case = np.mean(labels[i+10: i + 30])
        f.append(feature_for_case)
        l.append(label_for_case)
    features=np.array(f)
    labels=np.array(l)
	
	
    file = pd.read_csv('test_data.csv')
    data = file[['MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1']]
    data = np.array(data)
    data = sc.transform(data)
    L = len(data)
    dat = list()
    for i in range(L):
        feature_for_case = data[i*10: i*10 + 10]
        dat.append(feature_for_case)
    test = np.array(dat)
	
	
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(10, 7), return_sequences=False))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer="adam")
	
    history = model.fit(x=features, y=labels, batch_size=72, epochs=20,validation_split=0.3,shuffle=True)

    print(model.summary())

    predict = model.predict(test)
    predict = sc1.inverse_transform(predict)
    with open("predict.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["caseid", "midprice"])
        for i in range(142, len(predict)):
            writer.writerow([i+1, float(predict[i])])