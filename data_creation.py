import sqlite3
import numpy as np
import pickle
from data_preparation import DataPreparation
from windowing_scaling import WindowScale

class DataSelection:

    # @profile
    def __init__(self, all_data, year, save_path, use_all=True):
        self.data = all_data.loc[all_data["Timestamp"].dt.year == year].dropna()
        self.save_path = save_path
        self.range_tuples = self.find_good_ranges()
        self.final_data = np.empty((0,120,38))
        self.final_labels = np.empty((0, 120, 1))

        for gen in self.range_tuples:
            start, end = gen
            data = self.data.iloc[start:end+1, :]
            prep_data, prep_labels = self.data_prep(data, use_all)
            self.data_windowing(prep_data, prep_labels)

        self.data_save(self.final_data)
        self.data_save(self.final_labels)

    def data_prep(self, data, use_all):
        labels = self.generate_labels(data)
        data_object = DataPreparation(data, labels, use_all=use_all)
        data_object.select_variables()
        data_object.check_categorical()
        data_object.to_numpy()

        return data_object.array_data, data_object.labels

    def data_windowing(self, data, labels):
        # try:
        window_object = WindowScale(data, labels)
        windowed_data = window_object.windowed_data
        windowed_labels = window_object.windowed_labels

        self.final_labels = np.append(self.final_labels, windowed_labels, axis=0)
        self.final_data = np.append(self.final_data, windowed_data, axis=0)

        # except Exception as e:
        #     print("Windowing incomplete, error:")
        #     print(e)

    def data_save(self, data):
        with open(self.save_path, "wb") as file:
            pickle.dump(data, file)

    def generate_labels(self, data):
        label_map = {True: 1, False: 0}
        labels = ((data["BFLARE"] > 0) | (data["CFLARE"] > 0) |
                  (data["MFLARE"] > 0) | (data["XFLARE"] > 0)).replace(label_map)

        return labels

    def find_good_ranges(self):
        ranges = self.find_continuous_data()
        good_ranges = []

        for item in ranges:
            len_range = item[1]-item[0]
            if len_range >= 120:
                yield item[0], item[1]

    def find_continuous_data(self):
        continuous_range = []

        for i in range(len(self.data) - 2):
            if i == 0:
                diff = self.calc_time_diff(i)
                if diff == 12:
                    j = i + 1
                    while diff == 12:
                        diff = self.calc_time_diff(j)
                        j += 1
                    continuous_range.append((i, j-1))
                    yield i, j-1
            if i >= continuous_range[-1][1]:
                diff = self.calc_time_diff(i)
                if diff == 12:
                    j = i + 1
                    while (diff == 12 or diff == 0) and (j+1) < (len(self.data)-2):
                        diff = self.calc_time_diff(j)
                        j += 1
                    continuous_range.append((i, j - 1))
                    yield i, j-1

    def calc_time_diff(self, i):
        datetime_end = self.data.loc[:, "Timestamp"].iloc[i+1]
        datetime_start = self.data.loc[:, "Timestamp"].iloc[i]
        minutes_diff = (datetime_end - datetime_start).total_seconds() / 60.0

        return minutes_diff


if __name__ == "__main__":
    with open("all_data.pkl", "rb") as file:
        data = pickle.load(file)

    DataSelection(data, 2014, "./data_2014.pkl")
    DataSelection(data, 2015, "./data_2015.pkl")