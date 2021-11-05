import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data
filepath = 'oilprices.csv'
df = pd.read_csv(filepath)

# via numpy
s_dev_np = np.std(df['Price'])

mean = np.mean(df['Price'])

temp_list = []

for i in df['Price']:
    x = i - mean
    y = x * x
    temp_list.append(y)

var = np.mean(temp_list)

s_dev_manual = math.sqrt(var)

print(s_dev_manual,s_dev_np )

"""
training_data_length = int(len(df))


train_data, test_data = df[0:int(len(df) * 0.7)], df[int(len(df) * 0.7):]

training_data = train_data['Price'].values
testing_data = test_data['Price'].values


training_data, testing_data = training_data.reshape(-1, 1), testing_data.reshape(-1, 1)

# Transform the data for processing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(training_data)
scaled_test_data = scaler.fit_transform(testing_data)


# Configure the model params

timestep = 5
features = 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=timestep, batch_size=1)


lstm_model = Sequential()
lstm_model.add(LSTM(30, input_shape=(timestep, features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# lstm_model.summary()

# Train the Model
lstm_model.fit(generator, epochs=1)

# Save Model
# lstm_model.save(filepath='MY_LSTM_Model')

# Load Model
# lstm_model = load_model('MY_LSTM_Model')

# losses_lstm = lstm_model.history.history['loss']
# plt.figure(figsize=(12, 4))
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.xticks(np.arange(0, 21, 1))
# plt.plot(range(len(losses_lstm)), losses_lstm)
# plt.show()


lstm_predictions_scaled = list()

batch = scaled_test_data[:timestep]
current_batch = batch.reshape((1, timestep, features))

for i in range(len(testing_data) - 5):
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[scaled_test_data[i+5]]], axis=1)

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

print(lstm_predictions)

test_time = np.arange(start=0, stop=testing_data.size - 5)

# plt.plot(test_time, np.resize(scaled_test_data, scaled_test_data.size - 5), 'b')
# plt.plot(test_time, lstm_predictions_scaled, 'm')
# plt.show()

plt.plot(training_data)
plt.plot(testing_data[:testing_data.size - 5], lstm_predictions)
plt.plot(lstm_predictions)
plt.legend(['Train', 'Val', 'LSTM Predictions'], loc='lower right')
plt.show()
"""