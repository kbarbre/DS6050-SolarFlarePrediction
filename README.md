# DS6050-SolarFlarePrediction
## University of Virginia Masters of Data Science Program
DS 6050 Deep Learning Final Project on Solar Flare Prediction

Authors: Katelyn Barbre, Brian Nam, David Ackerman

## Goals

Our goal was to implement a Long-Short Term Memory (LSTM) model to predict the occurence of a solar flare 24 hours before it occurs. 

## Data

We are using the SWAN-SF database to complete our work. The open source data can be downloaded from here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EBCFKM

## Instructions for Repo

### Extracting the Data

First, the data_extraction.py file needs to be run to download data from the SWAN SF CSV files and save them into a SQLite3 database. Once the data has been extracted to a database, the InitialDataExploration.ipynb has code to pickle data from 2013-2018.

The data combination takes a few steps:

* Query solar flare data from SQLite3 database
* Convert Timestamp to datetime objects
* Drop duplicate rows
* Find missing timestamps in the solar flare data
* Query non-flare table in database for the missing timeframes
* Combine non-flare and flare data
* Save to a pickle file

### Data Selection

The data_creation.py combines several functions into one data call:

* Data preparation, to include selecting parameters, taking the average of duplicated non-flare timestamps, and converting to a numpy array.
* Data windowing and scaling, which normalizes, standardizes, and windows the data for use in the LSTM.
* Creating continuous windows of data to ensure the LSTM is learning on continuous data.

The output of running data_creation.py will be .npy files saved at the given location as well as the pickled normalization scaler and standardization scaler to be used on the test data.

### Model Building, Training, and Testing

The prediction_LSTM.py holds all the functions necessary to import data, build the LSTM model framework, train the model on training data, evaluate the model on test data, create predictions for a confusion matrix and classification metrics, and save the model for use later.

An example use of the file can be found in BaseModelPipeline.ipynb.


## Citations
Angryk, Rafal; Martens, Petrus; Aydin, Berkay; Kempton, Dustin; Mahajan, Sushant; Basodi, Sunitha; Ahmadzadeh, Azim; Xumin Cai; Filali Boubrahimi, Soukaina; Hamdi, Shah Muhammad; Schuh, Micheal; Georgoulis, Manolis, 2020, "SWAN-SF", https://doi.org/10.7910/DVN/EBCFKM, Harvard Dataverse, V1
