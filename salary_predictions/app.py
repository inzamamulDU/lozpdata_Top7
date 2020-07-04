from flask import Flask, render_template, request
import pandas as pd

import pickle
from features import FeatureMapper, SimpleTransform
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import re

#from test.predict import pre_processor
#from test.predict.py import pre_processor, feature_extractor, get_pipeline


app=Flask(__name__)

location = pd.read_csv('location_norm.csv')
location=list(location.values.tolist())
company = pd.read_csv('company_names.csv')
company = list(company[1:].values.tolist())
import pickle
import nltk
nltk.download('wordnet')
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
def pre_processor(text):
        return re.sub(r'[^0-9a-zA-Z]+', " ", text).lower()
    
def feature_extractor():
    features = [('FullDescription-Bag of Words', 'FullDescription', vectorizor),
                ('Title-Bag of Words', 'Title', vectorizor),
                ('LocationRaw-Bag of Words', 'LocationRaw', vectorizor),
                ('LocationNormalized-Bag of Words', 'LocationNormalized', vectorizor)]
    combined = FeatureMapper(features)
    return combined

def get_pipeline():
    features = feature_extractor()
    steps = [("extract_features", features),
             ("classify", RandomForestRegressor(n_estimators=50, 
                                                verbose=2,
                                                n_jobs=4,
                                                min_samples_split=30,
                                                random_state=3465343))]
    return Pipeline(steps)
filename = './test/data/random_forest.pickle'
regressor = pickle.load( open(filename,"rb"))



@app.route('/')
def home():
    
    return render_template('home.html',location=location,company = company)

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        my_job = {
        }
        my_job.LocationNormalized =  request.form['City']
        my_job.LocationRaw = my_job.LocationNormalized
        my_job.Company = request.form['Company']
        my_job.Title = request.form['JobTitle']
        my_job.FullDescription =  request.form['JobDesc']
        my_prediction = classifier.predict(data)
     
    
    return render_template('result.html',data = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)