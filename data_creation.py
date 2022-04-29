from pathlib import Path
import sqlite3
import numpy as np
import pickle
from data_preparation import DataPreparation
from windowing_scaling import WindowScale

def bytes_to_gb_conversion(num_bytes):
    num_gb = num_bytes / 1073741824
    print(f"Num GB used in data array: {num_gb}")
    return num_gb

class DataSelection:

    # @profile
    def __init__(self, all_data, year, save_path, num_variables=None, use_all=True, select_columns=None,
                 feature_selection=False, norm_scaler=None, stand_scaler=None):
        # self.data = all_data.loc[all_data["Timestamp"].dt.year == year].dropna()
        self.save_path = save_path
        p=Path(save_path)
        if not p.exists():
            p.mkdir()
        self.range_tuples = self.find_good_ranges(all_data)

        self.final_data = None
        self.final_labels = np.empty((0, 120, 1))

        if not feature_selection:
            if not num_variables:
                num_variables = 38
            self.final_data = np.empty((0, 120, num_variables))

        i = 0
        for gen in self.range_tuples:
            # if bytes_to_gb_conversion(self.final_data.size * self.final_data.itemsize) > 3.5:
            #     break
            start, end = gen
            data = all_data.loc[all_data["Timestamp"].dt.year == year].dropna().iloc[start:end+1, :]

            if norm_scaler:
                prep_data, prep_labels = self.data_prep(data, use_all=use_all, select_columns=select_columns,
                                                        feature_selection=feature_selection, norm_scaler=norm_scaler)
            else:
                prep_data, prep_labels = self.data_prep(data, use_all=use_all, select_columns=select_columns,
                                                        feature_selection=feature_selection)

            if not isinstance(prep_data, np.ndarray):
                continue

            if feature_selection and i == 0:
                num_variables = prep_data.shape[1]
                self.final_data = np.empty((0, 120, num_variables))

            if stand_scaler:
                self.data_windowing(prep_data, prep_labels, norm_scalar=norm_scaler, standard_scalar=stand_scaler)
            else:
                self.data_windowing(prep_data, prep_labels)
            bytes_to_gb_conversion(self.final_data.size * self.final_data.itemsize)

            i += 1

        # bytes_to_gb_conversion(self.final_data.size * self.final_data.itemsize)

        self.data_save(self.final_data, "data", year)
        self.data_save(self.final_labels, "labels", year)

    def data_prep(self, data1, use_all, select_columns, feature_selection, norm_scaler=None):
        if not norm_scaler:
            data_object = DataPreparation(data1, use_all=use_all, select_columns=select_columns,save_path=self.save_path)
        else:
            data_object = DataPreparation(data1, use_all=use_all, select_columns=select_columns, norm_scaler=norm_scaler,save_path=self.save_path)
        data_object.collapse_timestamp()

        if data_object.terminate:
            return False, False

        data_object.generate_labels()
        data_object.select_variables()
        data_object.check_categorical()
        data_object.normalize()
        if feature_selection:
            data_object.select_features()
        data_object.to_numpy()

        return data_object.array_data, data_object.labels

    def data_windowing(self, data2, labels, norm_scalar=None, standard_scalar=None):
        # try:
        window_object = WindowScale(data2, labels, norm_scalar=norm_scalar, standard_scalar=standard_scalar,save_path=self.save_path)
        windowed_data = window_object.windowed_data
        windowed_labels = window_object.windowed_labels

        self.final_labels = np.append(self.final_labels, windowed_labels, axis=0)
        self.final_data = np.append(self.final_data, windowed_data, axis=0)

        # except Exception as e:
        #     print("Windowing incomplete, error:")
        #     print(e)

    def data_save(self, data3, data_type, year):
        file_name = self.save_path + "/" + data_type + "_" + str(year) + ".npy"
        np.save(file_name, data3)

    def find_good_ranges(self, data5):
        ranges = self.find_continuous_data(data5)

        for item in ranges:
            len_range = item[1]-item[0]
            if len_range >= 120:
                yield item[0], item[1]

    def find_continuous_data(self, data6):
        continuous_range = []

        for i in range(len(data6) - 2):
            if i == 0:
                diff = self.calc_time_diff(data6, i)
                if diff == 12:
                    j = i + 1
                    while diff == 12:
                        diff = self.calc_time_diff(data6, j)
                        j += 1
                    continuous_range.append((i, j-1))
                    yield i, j-1
            if i >= continuous_range[-1][1]:
                diff = self.calc_time_diff(data6, i)
                if diff == 12:
                    j = i + 1
                    while (diff == 12 or diff == 0) and (j+1) < (len(data6)-2):
                        diff = self.calc_time_diff(data6, j)
                        j += 1
                    continuous_range.append((i, j - 1))
                    yield i, j-1

    def calc_time_diff(self, data7, i):
        datetime_end = data7.loc[:, "Timestamp"].iloc[i+1]
        datetime_start = data7.loc[:, "Timestamp"].iloc[i]
        minutes_diff = (datetime_end - datetime_start).total_seconds() / 60.0

        return minutes_diff


# if __name__ == "__main__":
#     with open("all_data.pkl", "rb") as file:
#         data = pickle.load(file)
#
#     DataSelection(data, 2014, "./")
#     DataSelection(data, 2015, "./")


if __name__ == "__main__":
    #Change this to stop main from doing things
    create_2014_data = True
    create_2015_data = False
    if create_2014_data:
        with open("all_data.pkl", "rb") as file:
            data = pickle.load(file)

        DataSelection(data, 2014, "./", feature_selection=True, use_all=False)
    if create_2015_data:
        with open("norm_scaler.pkl", "rb") as norm_file:
            normalization = pickle.load(norm_file)

        with open("stand_scaler.pkl", "rb") as stand_file:
            standard = pickle.load(stand_file)

        DataSelection(data, 2015,"./", norm_scaler=normalization, stand_scaler=standard)