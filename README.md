# Disaster Response Pipeline Project

The purpose of the project is to classify messages that have been recorded during a disease. The project has been developed as part of the Data Science Nanodegree Program by Udacity.

## Prerequisits

The sources of the repository are mostly in Python. Some parts related to the visualization are in HTML. Next to Python following pakets are needed to work with the project:

* sys
* pandas
* sqlalchemy
* sklearn
* numpy
* re
* pickle
* nltk
* json
* plotly
* flask

## Data Structure

To train the claffifier model two input files are needed:

* **data/disaster_messages.csv**: Includes the messages
* **data/disaster_categories.csv**: Includes the categories into which the messages can be divided 

## Scripts

* **data/process_data.py**: Reads the data from the csv's, cleans and stores it to a database
* **models/train_classifier.py**: Reads the data from the database, trains the classifier and saves the model as pkl file 
* **app/run.py**: Creates and runs the webpage including some visualizations 


## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/ to see the visualization

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

