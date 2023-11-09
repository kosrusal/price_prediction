import pandas as pd
import numpy as np
import get_prices as hist
import tensorflow as tf
from preprocessing import DataProcessing
import pandas_datareader.data as pdr
import yfinance as fix
import matplotlib.pyplot as plt
fix.pdr_override()

from keras.layers import Bidirectional 



k = 1
days_for_training = 5
Epochs = 50

start = "2014-01-06"
end = "2017-06-01"

start_pred = "2017-06-01"
end_pred = "2018-01-01"
ticker = "AAPL"

def back_test(strategy, seq_len, ticker, start_date, end_date, dim):
    """
    A simple back test for a given date period
    :param strategy: the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param seq_len: length of the days used for prediction
    :param ticker: company ticker
    :param start_date: starting date
    :type start_date: "YYYY-mm-dd"
    :param end_date: ending date
    :type end_date: "YYYY-mm-dd"
    :param dim: dimension required for strategy: 3dim for LSTM and 2dim for MLP
    :type dim: tuple
    :return: Percentage errors array that gives the errors for every test in the given date range
    """
    data = pdr.get_data_yahoo(ticker, start_date, end_date)
    stock_data = data["Adj Close"]

    print(len(stock_data))
    
    errors = []
    total_error=total_error = np.array(errors)
    print("!!!")
    #u=input()
    Predictions=[]
    Data=[]
    Dates = []
    for i in range((len(stock_data)//days_for_training)*days_for_training  - days_for_training - 1):
        x = np.array(stock_data.iloc[i: i + days_for_training]).reshape(1, days_for_training, 1) / k
        y = np.array(stock_data.iloc[i + days_for_training + 1]) / k
        predict = strategy.predict(x)
        print(x*k)
        print(y*k)
        #while predict == 0:
        #    predict = strategy.predict(x)
        error = (predict - y) / 100
        errors.append(error)
        total_error = np.array(errors)
        predict[0][0]*=k
        y*=k
        Predictions.append(predict)
        Data.append(y)
        print(f"Predict = {predict}, real = {y}")
        Dates.append(str(stock_data.index[i+days_for_training+1]))
    print(f"Average error = {total_error.mean()}")
    # If you want to see the full error list then print the following statement
    # print(errors)
    #print(Predictions)
    #print(Data)

    #Graphics
    """pred = []
    for i in range(len(Predictions)):
        pred.append(Predictions[i][0][0])
    df = pd.DataFrame()
    df.index = stock_data.index[15:]
    df['Predictions'] = pred
    df['Data'] = Data
        
    print(df.plot(figsize = (16, 16)))"""
    #Graphics
    pred = []
    for i in range(len(Predictions)):
      pred.append(Predictions[i][0][0])

    df = pd.DataFrame()
    df.index = Dates
    df['Predictions'] = pred
    df['Data'] = Data
    #print(Predictions)

    #(df.plot(figsize = (12, 12)))
    plt.plot(Dates,pred,'b',Dates,Data,'g')
    plt.legend(['Pred','Data'], loc=2)
    plt.show()

    






hist.get_stock_data(ticker, start_date=start, end_date=end)
process = DataProcessing("stock_prices.csv", 1)
#process.gen_test(days_for_training)
process.gen_train(days_for_training)

X_train = process.X_train.reshape(((int)(process.X_train.size/days_for_training), days_for_training, 1)) / k
Y_train = process.Y_train / k



    
#X_test = process.X_test.reshape((int)(process.X_test.size/days_for_training), days_for_training, 1) / k
#Y_test = process.Y_test / k

model = tf.keras.Sequential()
#First one
model.add(Bidirectional(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True), input_shape=(days_for_training, 1)))
model.add(Bidirectional(tf.keras.layers.LSTM(20, activation='relu', return_sequences=True)))
#Second one
"""
model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True, input_shape=(days_for_training, 1))) 
model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)) 
model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True)) 
model.add(tf.keras.layers.LSTM(25, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu')) 
model.add(tf.keras.layers.Dense(10, activation='relu'))
"""

model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

model.compile(optimizer="adam", loss="mse")

model.fit(X_train, Y_train, epochs=Epochs)

#model.evaluate(X_test, Y_test)



back_test(model, days_for_training, ticker, start_pred, end_pred, days_for_training)


