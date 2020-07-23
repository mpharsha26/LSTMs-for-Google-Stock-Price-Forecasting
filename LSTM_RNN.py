import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

training_set = dataset_train['Open'].values
training_set = training_set.reshape(len(training_set),1)

#applying feature scaling ( normalization ) on open stock price
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60 timesteps and 1 output
X_train, y_train = [], []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


#first row corresponds to stockprice at the 60th financial day of our trainset 
#all these values are the previous stock prices before t = 60
print(X_train[0])


#based on the above 60 stock prices, we'll train our future RNN to predict the stock price at time t+1
#stock price at time t+1 is nothing but this below one !! 
print(y_train[0])


#reshaping the array since keras expect a 3D array as input for RNN 
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the robust RNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout


#initialising the RNN as an instance of the sequential class
regressor = Sequential()


#adding the first LSTM layer and some Dropout regularisation
#dropout regularisation is added to prevent overfitting 
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(rate = 0.1))


#adding the 2nd LSTM layer with some dropout regularisation 
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.1))

#adding the 3rd LSTM layer along with some dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(rate = 0.1))


#adding our 4th LSTM layer and some dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(rate = 0.2))

#adding the output layer 
regressor.add(Dense(units = 1))

regressor.summary()


#attaching our RNN a powerful optimizer and a loss function
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


#fitting our model to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


#importing the real Google stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')

realStockPrice = dataset_test['Open'].values
realStockPrice = realStockPrice.reshape(len(realStockPrice),1)

#getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values

inputs = inputs.reshape(-1,1)

#scaling these inputs
inputs = sc.transform(inputs)

#creating a data structure with 60 timesteps for test set
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)


#these are the previous 60 stock prices of the first financial day of Jan, 2017
#basically this is the last 3 months data from the train set
X_test[0]

#reshaping into 3D 
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#predicting the stock prices of Jan, 2017 and scaling it back
predictedStockPrice = sc.inverse_transform(regressor.predict(X_test))

train_plot = sc.inverse_transform(regressor.predict(X_train))


#Visualising the predictions over the real stock prices
plt.plot(realStockPrice, color = 'red', label = 'Real Google Stock Price')
plt.plot(predictedStockPrice, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.ylim((0, 850))
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(realStockPrice, predictedStockPrice))
print(rmse)



