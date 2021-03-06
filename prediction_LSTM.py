from argparse import ArgumentError
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn import metrics
from pathlib import Path


class SolarLSTM:

    @classmethod
    def load_model(cls, save_path):
        cls_obj = cls(np.zeros((1, 1, 1)), None, save_path)
        cls_obj.model = keras.models.load_model(save_path)
        return cls_obj

    def __init__(self, solar_data, solar_labels, save_path, batch_size=64, tune=False,
                 units=(16, 128, 16), regularization=("early stopping", "dropout"), lr=0.001):
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

        self.batch_size = batch_size
        self.solar_train, self.solar_val = self.batch_prefetch_data(solar_data, solar_labels)
        self.parameters = solar_data.shape[2]
        # self.ensure_data_correctness(self.solar_data)
        # self.solar_labels = solar_labels
        self.tuning_pipeline = tune
        self.units_min = units[0]
        self.units_max = units[1]
        self.units_step = units[2]
        self.regularization = regularization
        self.adam_lr = lr
        self.save_path = save_path
        self.model = None
        self.callbacks = []  # Calback signals for training
        self.cutoff = 0.65  # Probability cutoff for binary class prediction

    def ensure_data_correctness(self, data):
        if not isinstance(data, np.ndarray):
            raise TypeError("Data needs to be input as numpy array")
        try:
            data.shape[2]
        except IndexError as e:
            raise IndexError("Data needs to be input as windowed data")
        if data[0][0][0] <= -1 or data[0][0][0] >= 1:
            raise ValueError("Data needs to be scaled between -1 and 1")

    def batch_prefetch_data(self, data, labels=None, split_data=True):
        shuffle_buffer = 100
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((data, labels)).shuffle(shuffle_buffer)
        else:  # Just prepare the data
            dataset = tf.data.Dataset.from_tensor_slices(data)
        cardinality = tf.data.experimental.cardinality(dataset).numpy()
        if split_data:
            train_dataset = dataset.take(cardinality * .8)
            val_dataset = dataset.skip(cardinality * .8)
            return train_dataset.batch(self.batch_size, drop_remainder=True).prefetch(2), \
                   val_dataset.batch(self.batch_size, drop_remainder=True).prefetch(2)
        else:
            return dataset.batch(self.batch_size, drop_remainder=True).prefetch(2)  # 1 return vs 2 if split_data

    def build_model(self):
        # What args we want for the callbacks
        # Define Adam optimizer: default settings for Adam are the same as our default settings
        opt = keras.optimizers.Adam(learning_rate=self.adam_lr)

        # Set loss function
        # If perfromance not great, try adding from_logits=True to BinaryCrossentropy
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

        # Add callbacks
        p = Path(self.save_path)
        if p.suffix:  # If has an extension (can't use is_dir incase not made yet)
            p = p.parent
        chkpt_path = p.joinpath('model_checkpoints')  # TODO We may need to os.mkdir
        # TODO decide what metrics/params to use here
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=chkpt_path, save_best_only=True),
        ]
        if not self.tuning_pipeline:
            model = keras.Sequential()
            model.add(
                # keras.layers.LSTM(units=16, batch_input_shape=(self.batch_size, 120, 38), stateful=True, return_sequences=True)
                keras.layers.LSTM(units=16, batch_input_shape=(self.batch_size, 120, self.parameters), stateful=True,
                                  return_sequences=True)
            )
            model.add(
                keras.layers.LSTM(units=16, return_sequences=True, stateful=True,
                                  batch_input_shape=(self.batch_size, 120, self.parameters))
            )

            # If dropout was included, add dropout layer
            if "dropout" in self.regularization:
                model.add(keras.layers.Dropout(rate=0.2))
            model.add(keras.layers.Dense(1, activation="sigmoid"))

            model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])

            self.model = model

    def _build_tuned_model(self, hp):
        # Creating keras tuner object
        # What args we want for the callbacks
        # Define Adam optimizer: default settings for Adam are the same as our default settings
        opt = keras.optimizers.Adam(learning_rate=self.adam_lr)

        # Set loss function
        # If perfromance not great, try adding from_logits=True to BinaryCrossentropy
        loss = keras.losses.BinaryCrossentropy(from_logits=True)

        # Create two-layer LSTM model with binary output
        model = keras.Sequential()
        model.add(
            keras.layers.LSTM(units=hp.Int("units", min_value=self.units_min,
                                           max_value=self.units_max, step=self.units_step),
                              # batch_input_shape=(self.batch_size, 120, 38), stateful=True,return_sequences=True)
                              batch_input_shape=(self.batch_size, 120, self.parameters), stateful=True,
                              return_sequences=True)  # , recurrent_dropout=0.2)
        )
        model.add(
            keras.layers.LSTM(units=hp.Int("units", min_value=self.units_min,
                                           max_value=self.units_max, step=self.units_step),
                              # batch_input_shape=(self.batch_size, 120, 38), stateful=True,return_sequences=True)
                              batch_input_shape=(self.batch_size, 120, self.parameters), stateful=True,
                              return_sequences=True)  # , recurrent_dropout=0.2)
        )
        # If dropout was included, add dropout layer
        if "dropout" in self.regularization:
            model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(1, activation="sigmoid"))

        model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "mse", "mae"])
        return model

    def save_model(self, model):
        model.save(self.save_path, save_format="h5")

    def fit(self, make_plots=True):
        if self.model is None:
            print('Model not found\nBuilding Model...')
            self.build_model()
        # TODO DETERMINE IF VALIDATION SPLIT,validation_split=0.1)
        if self.tuning_pipeline:
            # see keras tuners https://www.tensorflow.org/tutorials/keras/keras_tuner
            # We can't use val_accuracy when we don't have a split
            tuner = kt.BayesianOptimization(self._build_tuned_model, objective='val_accuracy', max_trials=10)
            tuner.search(self.solar_train, epochs=50, callbacks=self.callbacks, validation_data=self.solar_val)
            best_hyper = tuner.get_best_hyperparameters(num_trials=1)[0]
            print('Hyper Tuning Complete')
            self.model = tuner.hypermodel.build(best_hyper)
        history = self.model.fit(self.solar_train, epochs=50,
                                 callbacks=self.callbacks, validation_data=self.solar_val)
        if make_plots:
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)
            plt.title('Model Learning Performance')
            plt.show()
        # print('Model Fitting Done')
        # TODO maybe add some metrics or plots here?
        return history

    def evaluate(self, dataset):
        return self.model.evaluate(dataset)

    def predict(self, dataset):
        return self.model.predict(dataset, batch_size=self.batch_size)

    def confusion_from_logits(self, logits, true_labels, make_plots):
        preds = self._class_from_logits(logits)
        true_labels = true_labels.flatten()
        matrix = metrics.confusion_matrix(true_labels, preds, labels=[0, 1])
        if make_plots:
            self.plot_confusion_matrix(matrix)
        return matrix

    def predict_conf_matrix(self, dataset, labels, make_plots=True):
        logits = self.predict(dataset)
        return self.confusion_from_logits(logits, labels, make_plots)

    def plot_confusion_matrix(self, matrix):
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
        disp.plot(values_format='d')
        return disp  # Can modify the plot using this returned handle

    def view_prediction_distributions(self, true_labels, data=None, logits=None, make_plots=True):
        """
        Separates the distributions for true positves and false positves
        Will accept either data and do the prediction, or the already predicted logits
        """
        if data is None and logits is None:
            raise ArgumentError('Atleast one of data or logits must be specified')
        elif data:
            logits = self.predict(data)
        flat_logits = logits.flatten()
        pred_class_labels = self._class_from_logits(logits)
        true_class_labels = true_labels.flatten()
        pred_true_idx = np.where(pred_class_labels == 1)[0]
        actuals = true_class_labels[pred_true_idx]
        pred_logits = flat_logits[pred_true_idx]
        true_positive_logits = pred_logits[np.where(actuals == 1)[0]]
        false_positive_logits = pred_logits[np.where(actuals != 1)[0]]
        if make_plots:
            plt.boxplot(np.array([true_positive_logits, false_positive_logits]),
                        labels=['True Positives', 'False Positives'])
            plt.title('Boxplot of Probabilities for Solar Flare Predictions')
            plt.ylabel('Logit Prob')
        return (true_positive_logits, false_positive_logits)

    def _class_from_logits(self, logits):
        return np.where(logits.flatten() > self.cutoff, 1, 0)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def calculate_statistics(self, conf_matrix):
        tp = conf_matrix[1][1]
        fp = conf_matrix[1][0]
        tn = conf_matrix[0][0]
        fn = conf_matrix[0][1]

        # Precision - TP/(TP+FP)
        precision = tp / np.sum([tp, fp])

        # Recall - TP/(TP+FN)
        recall = tp / np.sum([tp, fn])

        # False Alarm Rate/Probability of False Alarm - FP/(FP+TN)
        far = fp / np.sum([fp, tn])

        # Heidke Skill Score (HSS) - 2*(TP * TN - FP * FN) / (TP+FN)*(FN+TN)+(TP+FP)*(FP+TN)
        hss = (2 * ((tp * tn) - (fp * fn))) / (((tp + fn) * (fn + tn)) + ((tp + tn) * (fp + tn)))

        return pd.DataFrame({"precision": precision, "recall": recall, "far": far, "hss": hss}, index=[0])
