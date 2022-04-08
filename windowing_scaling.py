import numpy as np
from sklearn.preprocessing import StandardScalar
from sklearn.preprocessing import MinMaxScalar


class WindowScale:

    def __init__(self, data, label, norm_scalar=None, standard_scalar=None, window_size=120):

        self.raw_data = data
        self.raw_label = label
        self.window_size = window_size

        # Check if a pre-fit normalization scalar was passed (for test data)
        if not norm_scalar:
            self.normalization_scalar = None
        else:
            self.normalization_scalar = norm_scalar

        # Check if a pre-fit standardization scalar was passed (for test data)
        if not standard_scalar:
            self.standardization_scalar = None
        else:
            self.standardization_scalar = standard_scalar

        self.data = self.normalize()
        self.windowed_data, self.windowed_labels = self.window()

    def normalize(self):
        """
        Function to normalize all data to the range of (-1, 1)
        :return: Transformed data
        """
        self.normalization_scaler = MinMaxScalar((-1, 1))
        self.normalization_scaler.fit(self.raw_data)

        return self.normalization_scaler.transform(self.raw_data)

    def window(self):
        """
        Function to normalize data, window it based on the passed constructor
        argument, and return
        :return: Windowed data and windowed labels
        """

        ######################################################
        # Windowing observation data
        ######################################################
        max_index = self.data.shape[0] - self.window_size

        """
        Found below method in the following article:
        https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
        """
        data_windows = (
                # expand_dims are used to convert a 2D array to 3D array.
                np.expand_dims(np.arange(self.window_size), 0) +
                np.expand_dims(np.arange(max_index + 1), 0).T
        )

        ######################################################
        # Windowing label data
        ######################################################
        end_index = self.raw_label.shape[0] - self.window_size

        labels = self.raw_label[self.window_size:end_index, :]
        labels = np.repeat(labels, self.window_size, axis=0)

        label_windows = (
                # expand_dims are used to convert a 2D array to 3D array.
                np.expand_dims(np.arange(self.window_size), 0) +
                np.expand_dims(np.arange(end_index + 1), 0).T
        )

        return self.data[data_windows], label_windows

    def standardize(self):
        pass

