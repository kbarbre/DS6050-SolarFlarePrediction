import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt


class SolarLSTM:

    def __init__(self, solar_data, solar_labels, save_path, tune=False, units=(64, 512, 32),
                 regularization=("early stopping", "dropout"), lr=0.001):
        """
        Constructor for Solar Flare Prediction pipeline.

        :param solar_data: Normalized data in numpy array
        :param tune: Whether or not to use the tuning pipeline. Default is True.
        :param units: Number of units to use within an LSTM layer. Default tuning
                      parameters are min=64, max=512, step=32.
        :param layers: Number of LSTM layers in the model. Default of 2.
        :param regularization: List of strings for early stopping techniques to use.
                               Currently only supports early stopping and dropout.
        :param lr: Learning rate for the Adam optimizer. Default of 0.001.
        """

        self.solar_data = solar_data
        self.ensure_data_correctness()
        self.solar_labels = solar_labels
        self.tuning_pipeline = tune
        self.units_min = units[0]
        self.units_max = units[1]
        self.units_step = units[2]
        self.regularization = regularization
        self.adam_lr = lr
        self.save_path = save_path

    def ensure_data_correctness(self):
        if isinstance(self.solar_data, pd.DataFrame):
            raise TypeError("Data needs to be input as numpy array")
        try:
            self.solar_data.shape[2]
        except IndexError as e:
            raise IndexError("Data needs to be input as windowed data")
        if self.solar_data[0][0][0] < -1 or self.solar_data[0][0][0] > 1:
            raise ValueError("Data needs to be scaled between -1 and 1")

    def build_model(self):
        # Define Adam optimizer: default settings for Adam are the same as our default settings
        opt = keras.optimizers.Adam(learning_rate=self.adam_lr)

        # Set loss function
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

        # TODO: Add keras.callbacks.ModelCheckpoint()
        # TODO: Add keras.callbacks.EarlyStopping()

        if not self.tuning_pipeline:
            model = keras.Sequential(
                keras.layers.LSTM(128),
                keras.layers.LSTM(128)
            )
            # If dropout was included, add dropout layer
            if "dropout" in self.regularization:
                model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.Dense(2, activation="sigmoid"))

            model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])

            return model
        else:
            model = self.build_tuned_model(opt, loss)
            return model

    def build_tuned_model(self, opt, loss):
        # Creating keras tuner object
        hp = kt.HyperParameters()

        # Create two-layer LSTM model with binary output
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(units=hp.Int("units", min_value=self.units_min,
                                           max_value=self.units_max, step=self.units_step),
                              activation="tanh")
        )
        model.add(
            keras.layers.LSTM(units=hp.Int("units", min_value=self.units_min,
                                           max_value=self.units_max, step=self.units_step),
                              activation="tanh")
        )
        # If dropout was included, add dropout layer
        if "dropout" in self.regularization:
            model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(2, activation="sigmoid"))

        model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])

        return model

    def save_model(self, model):
        if not self.tuning_pipeline:
            model.save(self.save_path)
        else:
            model.get_best_model().save()

    def fit(self):
        # TODO: Complete fit function. There are small differences in the fit for HyperParameter tuned model
        #       and regular tuned model
        pass

