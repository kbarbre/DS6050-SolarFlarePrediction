import pandas as pd
import time
from datetime import datetime
import os
import sqlite3


class TimeLog:
    ####################################
    # Running time calculation utility
    ####################################
    # Usage :
    # runTime = timeLog("Test")
    # runTime.beginProcess()
    # runTime.endProcess()
    # Created by:
    # Srinivasa (spc6ph@virginia.edu)
    # Abby (aeb4rv@virginia.edu)
    def __init__(self, procname=""):
        self.beginTime = time.time()
        self.endTime = time.time()
        self.procName = procname

    def begin_process(self):
        self.beginTime = time.time()
        print(datetime.now().strftime("%D %H:%M:%S"), " : Begin Process...", self.procName)

    def end_process(self):
        self.endTime = time.time()
        print(datetime.now().strftime("%D %H:%M:%S"), " : End Process...", self.procName)
        print(f"Total runtime for {self.procName} : {self.endTime - self.beginTime} seconds")


class DataExtractor:

    def __init__(self, sql_dir):
        self.db_file = sql_dir
        self.conn = None

    def df_to_sql(self, df, name):
        """
        Function to create a database connection, add the data to the
        database, and then close the database connection
        :param df: Dataframe to add to database
        :param name: Name of the table in the database
        :return: Nothing
        """

        conn = sqlite3.connect(self.db_file)
        df.to_sql(name, con=conn, if_exists='append', chunksize=10000)
        conn.close()


if __name__ == "__main__":

    flare_files = [file for file in os.listdir("E:/DS6050-SolarFlare/FL/")]
    non_flare_files = [file for file in os.listdir("E:/DS6050-SolarFlare/NF/")]

    t = TimeLog("Saving flares to sql database...")

    t.begin_process()
    for file in flare_files:
        extractor = DataExtractor("E:/DS6050-SolarFlare/solar_flare.db")
        data = pd.read_csv("E:/DS6050-SolarFlare/FL/" + file, sep="\t")
        extractor.df_to_sql(data, "flares")

    t.end_process()

    t2 = TimeLog("Saving non-flares to sql database...")

    t2.begin_process()
    for file in non_flare_files:
        extractor = DataExtractor("E:/DS6050-SolarFlare/solar_flare.db")
        data = pd.read_csv("E:/DS6050-SolarFlare/NF/" + file, sep="\t")
        extractor.df_to_sql(data, "non-flares")

    t2.end_process()
