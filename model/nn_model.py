import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten, LSTM, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from model.base_model import *

COVID_DAILY_URL = 'https://covid.ourworldindata.org/data/ecdc/new_cases.csv'
COVID_TOTAL_URL = 'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'

df = pd.read_csv(COVID_DAILY_URL)
df2 = pd.read_csv(COVID_TOTAL_URL)


daily_cases = df["United States"]
cumulative_cases = df2["United States"]
date = df["date"]

class modelNN(baseModel):
    def __init__(self):
        self.model = None

    def data_prep(self, window_size):
        temp_input = []
        temp_output = []
        for t in range(len(daily_cases)-window_size): #replaced with daily_cases
            x_input = daily_cases[t:t+window_size]
            temp_input.append(x_input)
            y_output = daily_cases[t+window_size]
            temp_output.append(y_output)

        input = np.array(temp_input).reshape(-1, window_size, 1)
        output = np.array(temp_output)
        return input, output

    def build_model(self,input, output):
        self.model = Sequential()
        self.model.add(LSTM(units=200, input_shape=(7, 1), return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(units=200))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(units=75))

        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=0.05)
        )

        history = self.model.fit(
            input[:-1], output[:-1], ## change back to input, output,
            batch_size=128,
            epochs=100000,
        )

        plt.plot(history.history['loss'])
        plt.show()

        return self.model

    def model_prediction(self,predict_set):
        # predict on the data
        for i in range(7):
            temp_predicted_cases = self.model.predict(
                predict_set,
            )
            predict_set = np.reshape(predict_set, (7,))
            predict_set = np.append(predict_set, temp_predicted_cases[0, 0])
            predict_set = np.delete(predict_set, 0)
            predict_set = np.reshape(predict_set, (-1, 7, 1))
        predict_set = predict_set.flatten()

        return predict_set
