from flask import Flask, render_template, request
import pandas as pd

import pickle


app=Flask(__name__)

location = pd.read_csv('location_norm.csv')
location=list(location.values.tolist())
company = pd.read_csv('company_names.csv')
company = list(company[1:].values.tolist())

filename = './test/data/random_forest.pickle'
classifier = pickle.load( open(filename,"rb"))

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