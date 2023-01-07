from marysroom.components import layers
from marysroom.models import NN
import numpy as np
import cupy as cp
import pandas as pd

import plotly.graph_objects as go


def load_coin_data(filename, *,
                   train_range=('2014-01-01', '2020-01-01'), test_range=('2020-01-01', '2022-01-01'),
                   n_inputs=1, n_outputs=1, value_col='value', date_col='date',
                   ):

    df = pd.read_csv(filename)
    df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(by=[date_col])
    df["pct_change"] = df[value_col].pct_change()
    df = df.fillna(0)

    train_range = pd.date_range(train_range[0], train_range[1], freq='H')
    train_data = df.loc[df[date_col].isin(train_range)]["pct_change"]
    train_data = train_data.to_numpy()

    test_range = pd.date_range(test_range[0], test_range[1], freq='H')
    test_data = df.loc[df[date_col].isin(test_range)]["pct_change"]
    test_data = test_data.to_numpy()

    dp_size = n_inputs + n_outputs

    X_train = []
    y_train = []
    for i in range(len(train_data)-dp_size):
        X_train.append(train_data[i:i+n_inputs])
        y_train.append(train_data[i+n_inputs: i+dp_size])

    X_train = np.stack(X_train, axis=0)
    y_train = np.stack(y_train, axis=0)

    X_test = []
    y_test = []
    for i in range(len(test_data)-dp_size):
        X_test.append(test_data[i:i+n_inputs])
        y_test.append(test_data[i+n_inputs: i+dp_size])

    X_test = np.stack(X_test, axis=0)
    y_test = np.stack(y_test, axis=0)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = load_coin_data(
        './Trade_Data/BTC_perday.csv',
        train_range=('2014-01-01', '2019-01-01'),
        test_range=('2019-01-01', '2021-12-31'),
        value_col='24h Open (USD)', date_col='Date',
        n_inputs=30, n_outputs=1,
    )

    # Initialize Neural Net model and pass it the array library to use.
    # numpy can be swapped for cupy to run on GPU. It defaults to numpy
    net = NN(ap=np)

    # # Uncomment to load a saved model
    # net.load_net(path='./Trained_Models/test_models/100.pickle', filename='test_net')

    # Add layers to the Neural Net model
    net.add_layer(layers.Dense(n_ins=30, n_outs=15, act='sigmoid'))
    net.add_layer(layers.Dense(n_ins=15, n_outs=1, act='sigmoid'))

    # Fit to the train data
    net.fit(X_train, y_train, lr=0.001,
            batch_size=1000,                        # num datapoints to train at once
            epochs=100,                             # num iterations through dataset
            display=10,                             # progress display frequency
            loss='mse',                             # loss function
            opt='nadam',                            # optimizing algorithm
            decayV=0.9, decayG=0.8, epsilon=1e-8,   # optimizer settings
            checkpoint_name='test_models/',         # folder name to store checkpoints in
            checkpoint_path='./Trained_Models/',    # where to store checkpoints
            checkpoints=10,                         # how frequently to save checkpoints
            onehot=False,                           # weather to use one hot enconding
            )

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=net.loss_curve, mode='lines'))
    fig.show()

    print(net.predict(X_test[:2]))

    # # Uncomment to save the model
    # net.save_net(path='./Trained_Models/test_models/', filename='100.pickle')
