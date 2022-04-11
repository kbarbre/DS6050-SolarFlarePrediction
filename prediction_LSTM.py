import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

from pathlib import Path
class SolarLSTM:

    @classmethod 
    def load_model(cls,save_path):
        cls_obj=cls(np.zeros((1,1,1)),None,save_path)
        cls_obj.model = keras.models.load_model(save_path)
        return cls_obj

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
        # self.ensure_data_correctness(self.solar_data)
        self.solar_labels = solar_labels
        self.tuning_pipeline = tune
        self.units_min = units[0]
        self.units_max = units[1]
        self.units_step = units[2]
        self.regularization = regularization
        self.adam_lr = lr
        self.save_path = save_path
        self.model = None
        self.callbacks = [] #Calback signals for training

    def ensure_data_correctness(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data needs to be input as numpy array")
        try:
            data.shape[2]
        except IndexError as e:
            raise IndexError("Data needs to be input as windowed data")
        if data[0][0][0] <= -1 or data[0][0][0] >= 1:
            raise ValueError("Data needs to be scaled between -1 and 1")

    def build_model(self):
        #What args we want for the callbacks
        # Define Adam optimizer: default settings for Adam are the same as our default settings
        opt = keras.optimizers.Adam(learning_rate=self.adam_lr)

        # Set loss function
        #If perfromance not great, try adding from_logits=True to BinaryCrossentropy
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

        #Add callbacks
        p=Path(self.save_path)
        if p.suffix:#If has an extension (can't use is_dir incase not made yet)
            p = p.parent
        chkpt_path = p.joinpath('model_checkpoints')#TODO We may need to os.mkdir
        #TODO decide what metrics/params to use here
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2),
            tf.keras.callbacks.ModelCheckpoint(filepath=chkpt_path,save_best_only=True),
        ]
        if not self.tuning_pipeline:
            model = keras.Sequential()
            model.add(
                keras.layers.LSTM(units=16, batch_input_shape=(1000, 120, 38), stateful=True, return_sequences=True)
            )
            model.add(
                keras.layers.LSTM(units=16, return_sequences=True, stateful=True, batch_input_shape=(1000, 120, 38))
            )

            # If dropout was included, add dropout layer
            if "dropout" in self.regularization:
                model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.Dense(1, activation="sigmoid"))

            model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])
        else:
            model = self._build_tuned_model(opt, loss)
        self.model = model

    def _build_tuned_model(self, opt, loss):
        # Creating keras tuner object
        hp = kt.HyperParameters()

        # Create two-layer LSTM model with binary output
        model = keras.Sequential()
        model.add(
            keras.layers.InputLayer((self.solar_data.shape[0], self.solar_data.shape[1], self.solar_data.shape[2]))
        )
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
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])
        return model

    def save_model(self, model):
        if not self.tuning_pipeline:
            model.save(self.save_path)
        else:
            model.get_best_model().save()

    def fit(self):
        if self.model is None:
            print('Model not found\nBuilding Model...')
            self.build_model()
        #TODO DETERMINE IF VALIDATION SPLIT,validation_split=0.1)
        if self.tuning_pipeline:
            #see keras tuners https://www.tensorflow.org/tutorials/keras/keras_tuner
            #We can't use val_accuracy when we don't have a split
            tuner = kt.BayesianOptimization(self.model,objective='accuracy',max_trials=10)
            tuner.search(self.solar_data, self.solar_labels, epochs=50, validation_split=0.0, callbacks=self.callbacks)
            best_hyper = tuner.get_best_hyperparameters(num_trials=1)[0]
            print('Hyper Tuning Complete')
            self.model = tuner.hypermodel.build(best_hyper)
        history = self.model.fit(self.solar_data, self.solar_labels, epochs=50, callbacks=self.callbacks)
        # print('Model Fitting Done')
        #TODO maybe add some metrics or plots here?

    def evaluate(self,new_data,new_labels):
        #TODO determine if any optional args here
        return self.model.evaluate(new_data,new_labels)