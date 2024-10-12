from flask import Flask,render_template,url_for,redirect,flash
from form import InputForm
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'
pipe = pickle.load(open('jaipur.pkl','rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/Titanic_Survival_Prediction',methods=['GET','POST'])
def predict():
    form = InputForm()
    message=''
    if form.validate_on_submit():
        Pclass = form.Pclass.data
        Sex = form.Sex.data
        Age = form.Age.data
        Fare = form.Fare.data
        Embarked = form.Embarked.data
        Family_Members = form.Family_Members.data
        df = pd.DataFrame({'Pclass':[Pclass],'Sex':[Sex],'Age':[Age],'Fare':[Fare],'Embarked':[Embarked],'Family Members':[Family_Members]})

        output = pipe.predict(df)[0]
        chances=""
        if output==1:chances='Survive'
        else:chances='Not Survive'
        message = 'The passenger will {a}'.format(a=chances)
    else:
        pass
    return render_template('predict.html',form=form,output=message)


if __name__ == "__main__":
    app.run(debug=True)

