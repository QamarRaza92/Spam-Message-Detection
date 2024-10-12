from flask import Flask,render_template,redirect,url_for,flash 
import pandas as pd 
import numpy as np 
import pickle 
from form import InputForm




classifier = pickle.load(open('SMS Spam Classifier/RandomForest.pkl','rb'))
vectorizer = pickle.load(open('SMS Spam Classifier/Vectorizer.pkl','rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html',title='Home')

@app.route('/predict', methods = ['GET','POST'])
def predict():
    output = ''
    form = InputForm()
    if form.validate_on_submit():
        message = form.message.data

        df = pd.DataFrame(vectorizer.transform([message]).toarray())
        result = classifier.predict(df)
        if result == 1:
            output = f"Spam"
        else:
            output = f"Not Spam"
    
    else:
        flash("")
    return render_template('predict.html',title='predict',form=form,output=output)


if __name__ == '__main__':
    app.run(debug=True)
