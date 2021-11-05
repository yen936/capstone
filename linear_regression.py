import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing


# Load data
filepath = 'oilprices.csv'
df = pd.read_csv(filepath)


# Normalizing
price_norm = preprocessing.scale(df['Price'].values)
date_norm = preprocessing.scale(np.arange(df['Date'].size))

# Loads numpy array into tensor
x_train = torch.tensor(date_norm)  # array of number of days
y_train = torch.tensor(price_norm)

# formatting and reshaping
x_train, y_train = x_train.type(torch.FloatTensor), y_train.type(torch.FloatTensor)
x_train, y_train = x_train.reshape(-1, 1), y_train.reshape(-1, 1)


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 in feature and 1 out feature

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


num_epochs = 1000
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Oil Prices')
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.legend()
plt.title(label="Linear Regression of Oil Prices")
plt.show()
