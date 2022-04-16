import numpy as np
import sklearn.preprocessing as preproc
import pickle


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

        # Series of steps to normalize, window, and standardize the data
        self.windowed_data, self.windowed_labels = self.window()
        self.standardize()
        self.normalize()

    def normalize(self):
        """
        Function to normalize all data to the range of (-1, 1)
        :return: Transformed data
        """
        save=False

        if not self.normalization_scalar:
            self.normalization_scalar = preproc.MinMaxScaler((-1, 1))
            save=True

        reshape_data = np.reshape(self.windowed_data, (self.windowed_data.shape[0] * self.windowed_data.shape[1],
                                                       self.windowed_data.shape[2]))
        transformed_data = self.normalization_scalar.fit_transform(reshape_data)
        self.windowed_data = np.reshape(transformed_data, self.windowed_data.shape)

        if save:
            pickle.dump(self.normalization_scalar, open("norm_scaler.pkl", "wb"))


    def window(self):
        """
        Function to normalize data, window it based on the passed constructor
        argument, and return
        :return: Windowed data and windowed labels
        """

        ######################################################
        # Windowing observation data
        ######################################################
        max_index = self.raw_data.shape[0] - self.window_size

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

        labels = []

        for i in range(end_index+1):
            create_labels = self.raw_label[i:i+self.window_size-1]
            labels.append(np.full(1, np.max(create_labels)))

        labels = np.array(labels)
        labels = np.repeat(labels, self.window_size, axis=0)
        labels = labels.reshape((end_index+1, self.window_size, 1))
        # labels = np.expand_dims(labels, 0)
        # labels = labels.reshape((labels.shape[0], self.window_size, 1))

        # end_index = labels.shape[0] - self.window_size
        #
        # label_windows = (
        #         # expand_dims are used to convert a 2D array to 3D array.
        #         np.expand_dims(np.arange(self.window_size), 0) +
        #         np.expand_dims(np.arange(end_index + 1), 0).T
        # )

        return self.raw_data[data_windows], labels

    def standardize(self):
        """
        Function to standardize data for across each window, for each time step,
        and for every variable in that time step
        :return: Standardized, normalized, and windowed data
        """

        save = False

        if not self.standardization_scalar:
            self.standardization_scalar = preproc.StandardScaler()
            save=True

        for i in range(self.windowed_data.shape[1]):
            self.windowed_data[:, i, :] = self.standardization_scalar\
                                              .fit_transform(self.windowed_data[:, i, :], 1)

        if save:
            pickle.dump(self.standardization_scalar, open("stand_scaler.pkl", "wb"))


