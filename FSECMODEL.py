# Nicholas Mistry
import csv
import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import keras_tuner as kt

sns.set(style='whitegrid', context='notebook')

NUMBER_OF_COLUMNS = 0


class FileC:
    def __init__(self, file):
        self.file = file

    def getHeader(self):
        f = open(self.file)
        c = csv.reader(f)
        header = next(c)
        count = 0
        to_drop = []
        for i in header:
            if count > NUMBER_OF_COLUMNS:
                to_drop.append(i)
            count = count + 1

        return to_drop

    def getSomeColumns(self):
        df = pd.read_csv(self.file, low_memory=False)
        df.drop(self.getHeader(), inplace=True, axis=1)
        return df


class AutoEncoder(Model):

    def __init__(self, output_units, code_size=8):
        super().__init__()
        self.encoder = Sequential([
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(code_size, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(16, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderTuner(Model):

    def __init__(self, hp, output_units, code_size=8):
        super().__init__()
        dense_1_units = hp.Int('dense_1_units', min_value=16, max_value=72, step=4)
        dense_2_units = hp.Int('dense_2_units', min_value=16, max_value=72, step=4)
        dense_3_units = hp.Int('dense_3_units', min_value=16, max_value=72, step=4)
        dense_4_units = hp.Int('dense_4_units', min_value=16, max_value=72, step=4)
        dense_5_units = hp.Int('dense_5_units', min_value=16, max_value=72, step=4)
        dense_6_units = hp.Int('dense_6_units', min_value=16, max_value=72, step=4)

        self.encoder = Sequential([
            Dense(dense_1_units, activation='relu'),
            Dropout(0.1),
            Dense(dense_2_units, activation='relu'),
            Dropout(0.1),
            Dense(dense_3_units, activation='relu'),
            Dropout(0.1),
            Dense(code_size, activation='relu')
        ])
        self.decoder = Sequential([
            Dense(dense_4_units, activation='relu'),
            Dropout(0.1),
            Dense(dense_5_units, activation='relu'),
            Dropout(0.1),
            Dense(dense_6_units, activation='relu'),
            Dropout(0.1),
            Dense(output_units, activation='sigmoid')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


def buildModel(hp):
    model = AutoEncoderTuner(hp, 8)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        loss='msle',
        optimzer=Adam(learning_rate=hp_learning_rate),
    )
    return model




RATIO = 15
RANDOM_SEED = 30
np.random.seed(RANDOM_SEED)


def find_threshold(model, x_train_scaled):
    reconstructions = model().predict(x_train_scaled)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)

    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) \
                + np.std(reconstruction_errors.numpy())
    return threshold


def get_predictions(model, x_test_scaled, threshold):
    predictions = model().predict(x_test_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds


TIME_STEPS = 12463


def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


def main():
    df = pd.read_csv("data6.csv", header=None, low_memory=False)
    # print(FileC("data2.csv").getHeader())
    # df.columns = map(str.lower, df.columns)
    # df.rename(columns={'class': 'label'}, inplace=True)
    print(df.head())
    data = df
    TARGET = 8

    features = data.drop(TARGET, axis=1)
    target = data[TARGET]


    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    train_index = y_train[y_train == 1].index
    train_data = x_train.loc[train_index]

    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = min_max_scaler.fit_transform(train_data.copy())
    x_test_scaled = min_max_scaler.transform(x_test.copy())

    model = AutoEncoder(output_units=x_train_scaled.shape[1])
    model.compile(loss='msle', metrics=['mse'], optimizer='adam')

    history = model.fit(
        x_train_scaled,
        x_train_scaled,
        epochs=20,
        batch_size=512,
        validation_data=(x_test_scaled, x_test_scaled)
    )

    model.save('FSECModel')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('MSLE Loss')
    plt.legend(['loss', 'val_loss'])
    plt.show()
    threshold = find_threshold(model, x_train_scaled)
    print(f"Threshold method one: {threshold}")

    #preds = get_predictions(model, x_test_scaled, threshold)
    #print(accuracy_score(preds, y_test))
    #threshold_ = find_threshold(best_model, x_train_scaled)
    #preds_ = get_predictions(best_model, x_test_scaled, threshold_)
    #print("TUNED: " + str(accuracy_score(preds_, y_test)))


main()
