from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Bidirectional 
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
import yfinance as fin
fin.pdr_override()

import get_prices as hist
from preprocessing import DataProcessing

model_save_dir = ""

k = 1
days_for_training = 16
Epochs = 50

start = "2020-01-06"
end = "2023-06-01"

start_pred = "2023-06-01"
end_pred = "2023-09-01"
ticker = "AAPL"

def predict_n(model, input, count):
    predictions = list(input)
    for i in range(len(input) - days_for_training + count):
        pred = model.predict(np.array(predictions[i:i+days_for_training]).reshape(1, days_for_training, 1) / k)[0][0][0] * k
        if i + days_for_training >= len(predictions):
            predictions.append(pred)
    
    return predictions[-count:]

def test_predict_n(strategy, ticker, start_date, end_date, end_pred_date):
    stock_data = pdr.get_data_yahoo(ticker, start_date, end_date)["Adj Close"]
    result_stock_data = pdr.get_data_yahoo(ticker, end_date, end_pred_date)["Adj Close"]
    print(len(stock_data))
    print("!!!")

    predictions = predict_n(strategy, stock_data, len(result_stock_data))

    #Graphics
    plt.plot(result_stock_data.index, predictions, 'b', result_stock_data.index, result_stock_data, 'g')
    plt.legend(['pred','data'], loc=2)
    plt.show()



def back_test(strategy, ticker, start_date, end_date):
    """
    A simple back test for a given date period
    :param strategy: the chosen strategy. Note to have already formed the model, and fitted with training data.
    :param ticker: company ticker
    :param start_date: starting date
    :type start_date: "YYYY-mm-dd"
    :param end_date: ending date
    :type end_date: "YYYY-mm-dd"
    :type dim: tuple
    :return: none
    """
    stock_data = pdr.get_data_yahoo(ticker, start_date, end_date)["Adj Close"]

    print(len(stock_data))
    print("!!!")
    
    errors = []
    predictions = []
    data = []
    dates = []
    for i in range(len(stock_data) - days_for_training - 1):
        x = np.array(stock_data.iloc[i: i + days_for_training]).reshape(1, days_for_training, 1) / k
        y = np.array(stock_data.iloc[i + days_for_training]) / k
        predict = strategy.predict(x)[0][0]
        #print(x*k)
        #print(y*k)
        error = (predict - y) / 100
        errors.append(error)
        predictions.append(predict * k)
        data.append(y * k)
        #print(f"Predict = {predict}, real = {y}")
        dates.append(str(stock_data.index[i+days_for_training+1]))

    #print(f"Average error = {np.array(errors).mean()}")
    # If you want to see the full error list then print the following statement
    #print(errors)
    print(predictions[0][0])
    #print(data)
    return predictions[0][0]
    #Graphics
    plt.plot(dates, predictions, 'b', dates, data, 'g')
    plt.legend(['pred','data'], loc=2)
    plt.show()


#----------------------------------------------------------------! поменял директорию
def load_model(ticker):
    return tf.keras.models.load_model(model_save_dir + f"model_{ticker}.keras")

def save_model(model, ticker):
    model.save(model_save_dir + f"model_{ticker}.keras")

def create_model():
    model = tf.keras.Sequential()
    #First one
    model.add(Bidirectional(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True), input_shape=(days_for_training, 1)))
    model.add(Bidirectional(tf.keras.layers.LSTM(20, activation='relu', return_sequences=True)))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
    #Second one
    """
    model.add(tf.keras.layers.LSTM(200, activation='relu', return_sequences=True, input_shape=(days_for_training, 1))) 
    model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)) 
    model.add(tf.keras.layers.LSTM(50, activation='relu', return_sequences=True)) 
    model.add(tf.keras.layers.LSTM(25, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu')) 
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
    """
    model.compile(optimizer="adam", loss="mse")
    return model

def train_model(model, ticker, start_learn, end_learn):
    prices_filename = hist.get_stock_data(ticker, start_date=start_learn, end_date=end_learn)
    process = DataProcessing(prices_filename, 1)
    process.gen_train(days_for_training)

    X_train = process.X_train.reshape(((int)(process.X_train.size / days_for_training), days_for_training, 1)) / k
    Y_train = process.Y_train / k
    model.fit(X_train, Y_train, epochs=Epochs)


def main():
    model = create_model()
    train_model(model, ticker, start, end)
    #model = load_model(ticker)
    
    back_test(model, ticker, start_pred, end_pred)

    #test_predict_n(model, ticker, "2019-12-01", "2020-01-01", "2020-02-01")

    if int(input("save model? ")) != 0:
        save_model(model, ticker)

if __name__ == "__main__":
    main()
