import tensorflow as tf
import time
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

import numpy as np
from util import csv_to_dataset, history_points


# Dataset
stock = "AAPL_daily.csv"
ohlcv_histories, _, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset(stock)

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)


def plot_loss(fitted,string):
    plt.plot(fitted.history[string],lw=3,c="blue")
    plt.title("Loss Function")
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string])
    plt.grid(True)
    plt.show()

# Build Model
lstm_input = Input(shape=(history_points, 5), name='lstm_input')
x = LSTM(50, name='lstm_0')(lstm_input)
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
output = Activation('linear', name='linear_output')(x)

model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=10, min_lr=0.00001)

model.compile(optimizer=adam, loss='mse')

#Fit model with training values
start_time=time.time()
history=model.fit(x=ohlcv_train, y=y_train, batch_size=32, epochs=200, shuffle=True, validation_split=0.1)
end_time=time.time()

#Plot Loss functions
plot_loss(history, "loss")
plot_loss(history, "val_loss")

# Evaluate results
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories)
y_predicted = y_normaliser.inverse_transform(y_predicted)


assert unscaled_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(unscaled_y_test) - np.min(unscaled_y_test)) * 100
print("Scaled MSE: ",scaled_mse)
print("Total Execution Time: ",round(((end_time-start_time)/60),3),"minutes")

plt.figure(figsize=(16, 12))

start = 0
end = -1

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
plt.legend(['Real', 'Predicted'])
plt.title(stock)
plt.grid()
plt.show()

val_real = plt.plot(unscaled_y[start:end], label='real')
val_pred = plt.plot(y_predicted[start:end], label='predicted')
plt.legend(['Real', 'Predicted'])
plt.title(stock)
plt.grid()
plt.show()

#Save model
tf.keras.models.save_model(model,'basic_model.h5')
bm_json = model.to_json()
with open(("basic_model_json.json"),"w") as json_file:
    json_file.write(bm_json)