import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn import preprocessing, metrics
from pmdarima.arima import auto_arima, ndiffs



# Load data
filepath = 'oilprices.csv'
df = pd.read_csv(filepath)


# Plotting the auto-correlation of the data
'''
plt.figure()
pd.plotting.lag_plot(df['Price'], lag=3)
plt.title('Oil Price Stock - Autocorrelation plot with lag = 3')
plt.show()
'''


def plot_autocorrelation():
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    plt.title('Oil Price Autocorrelation plot')

    # The axis coordinates for the plots
    ax_idcs = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1)
    ]

    for lag, ax_coords in enumerate(ax_idcs, 1):
        ax_row, ax_col = ax_coords
        axis = axes[ax_row][ax_col]
        pd.plotting.lag_plot(df['Price'], lag=lag, ax=axis)
        axis.set_title(f"Lag={lag}")

    plt.show()

train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.7):]

training_data = train_data['Price'].values
test_data = test_data['Price'].values

kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")
# Estimated differencing term: 1



