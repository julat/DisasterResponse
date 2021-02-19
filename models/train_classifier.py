import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


def load_data(database_filepath):
    """
    Load the data from the database and define X and Y 

    Parameters:
		database_filepath (str): 	Path to messages csv

	Returns:
        Dataframe: 	Input data 
		DataFrame: 	Output data
		List:		Names of categories
    """
	
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponseTable',engine)
	
    X = df['message']
    Y = df.iloc[:,4:]
	
    category_names = Y.columns
	
    return X,Y,category_names


def tokenize(text):
    """
    Tokenizes, removes stop words and lemmatizes the input text 

    Parameters:
		database_filepath (str): 	Path to messages csv

	Returns:
        List: 	List of cleaned words

    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    words = word_tokenize(text)
       
    words = [w for w in words if w not in stopwords.words("english")]

    lemmed_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    
    return lemmed_words


def build_model():
    """
    Builds pipeline as well as parameters for grid search

	Returns:
        Model: 	List of cleaned words

    """
    pipeline_rfc = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
	
    parameters = {'clf__estimator__n_estimators': [50, 100, 200],
            'clf__estimator__min_samples_split': [2, 3, 4]}

    cv = GridSearchCV(pipeline_rfc, param_grid=parameters)
	
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict data and print metrics
    
    Parameters:
		model (): 			Path to messages csv
		X_test (): 			Test input data
		Y_test (): 			Test output data
		category_names (): 	Names of the categories

    """
	
    Y_pred = model.predict(X_test)
	
    i = 0
    for col in Y_test:
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))
        i = i + 1


def save_model(model, model_filepath):
    """
    Save model to file
    
    Parameters:
		model (): 	Path to messages csv
		model_filepath (str): Path where the model should be stored

    """
	
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()