import pandas as pd
import numpy as np
from windowing_scaling import WindowScale
import pickle


class DataPreparation:

    def __init__(self, data, use_all=False, select_columns=None):
        self.raw_data = data
        if not select_columns:
            self.columns = data.columns
        else:
            self.columns = select_columns
        self.use_all = use_all
        self.array_data = None
        self.labels = None
    
    def collapse_timestamp(self):
        self.collapsed_data = self.raw_data.groupby('Timestamp').mean()
        self.collapsed_data.reset_index(inplace = True)

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






