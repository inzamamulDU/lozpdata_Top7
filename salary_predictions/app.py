from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    data  ={
       'salaryup': 1200,
       'salarydown': 1800
        }
    return render_template('result.html',data = data)


if __name__ == '__main__':
	  app.run(debug=True)