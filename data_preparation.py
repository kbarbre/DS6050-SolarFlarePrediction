import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import math
import pickle
import sklearn.preprocessing as preproc
import pandas as pd


class DataPreparation:

    def __init__(self, data, use_all=False, select_columns=None, norm_scaler=None,save_path='./'):
        self.raw_data = data
        self.save_path = save_path
        if not select_columns:
            self.columns = data.columns
        else:
            self.columns = select_columns
        self.use_all = use_all
        if not norm_scaler:
            self.normalization_scalar = None
        else:
            self.normalization_scalar = norm_scaler
        self.array_data = None
        self.labels = None
        self.terminate = False
    
    def collapse_timestamp(self):
        self.collapsed_data = self.raw_data.groupby('Timestamp').mean()
        self.collapsed_data.reset_index(inplace = True)

        if len(self.collapsed_data) < 120:
            self.terminate = True

    def generate_labels(self):
        label_map = {True: 1, False: 0}
        labels = ((self.collapsed_data["BFLARE"] > 0) | (self.collapsed_data["CFLARE"] > 0) |
                  (self.collapsed_data["MFLARE"] > 0) | (self.collapsed_data["XFLARE"] > 0)).replace(label_map)

        self.labels = labels

    def select_variables(self):
        # Remove unnecessary columns
#         col_remove = ["Timestamp", "BFLARE_LABEL", "CFLARE_LABEL", "MFLARE_LABEL", "XFLARE_LABEL",
#                       "BFLARE_LOC", "BFLARE_LABEL_LOC", "CFLARE_LOC", "CFLARE_LABEL_LOC", "MFLARE_LOC",
#                       "MFLARE_LABEL_LOC", "XFLARE_LOC", "XFLARE_LABEL_LOC", "QUALITY", "IS_TMFI", "XR_MAX", "XR_QUAL"]
        col_remove = ["Timestamp", "BFLARE_LOC", "CFLARE_LOC", "MFLARE_LOC", "XFLARE_LOC", 
                      "QUALITY", "IS_TMFI", "XR_MAX", "XR_QUAL"]
#         new_data = self.raw_data.drop(col_remove, axis=1)
        new_data = self.collapsed_data.drop(col_remove, axis=1)
        if self.use_all:
            self.array_data = new_data
        else:
            for column in new_data.columns:
                if column not in self.columns:
                    new_data = new_data.drop(column, axis=1)
                else:
                    continue
                    # print("Column already removed in preprocessing...")
            self.array_data = new_data

    def select_features(self):
        k_num = math.floor(len(self.array_data.columns) / 3)  # select the top third based on the Chi Squared f-score
        #     fs = SelectKBest(score_func=chi2, k='all')
        fs = SelectKBest(score_func=chi2, k=k_num)

        if not isinstance(self.labels, pd.Series):
            self.labels = pd.Series(self.labels)
            print(isinstance(self.labels, pd.Series))

        fs.fit(self.array_data, list(self.labels.values))

        self.array_data = fs.transform(self.array_data)

    def normalize(self):
        """
        Function to normalize all data to the range of (0, 1)
        :return: Transformed data
        """
        save = False

        if not self.normalization_scalar:
            self.normalization_scalar = preproc.MinMaxScaler((0, 1))
            save = True

        all_col = self.array_data.columns
        self.array_data = self.normalization_scalar.fit_transform(self.array_data)
        self.array_data = pd.DataFrame(self.array_data, columns=all_col)

        if save:
            pickle.dump(self.normalization_scalar, open(self.save_path+"norm_scaler.pkl", "wb"))

    def check_categorical(self):
        for column in self.array_data.columns:
            if not isinstance(type(column), int) or isinstance(type(column), float):
                # print(f"{column} type: {type(column)}\nType incompatible with numpy arrays...\nAttempting to convert")
                try:
                    self.array_data[column] = self.array_data[column].astype(float)
                    # print("Successful")
                except Exception as e:
                    continue
                    # print(f"Unable to convert: {e}")

    def to_numpy(self):
        self.array_data = np.array(self.array_data)
        self.labels = np.array(self.labels)

# if __name__ == "__main__":
#     with open("all_data.pkl", "rb") as file:
#         data = pickle.load(file)
#
#     print("Data read in")
#
#     data = data.loc[data["Timestamp"].dt.year == 2014].dropna()
#
#     print("Data filtered")
#
#     test_object = DataPreparation(data=data, use_all=False)
#     test_object.collapse_timestamp()
#     test_object.generate_labels()
#     test_object.select_variables()
#     test_object.check_categorical()
#     test_object.normalize()
#     test_object.select_features()
#     test_object.to_numpy()
#
#     print(type(test_object.array_data))
#     print(test_object.array_data.shape)


