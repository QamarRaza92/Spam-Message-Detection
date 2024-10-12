from flask_wtf import FlaskForm
from wtforms import SelectField,IntegerField,FloatField,SubmitField
from wtforms.validators import DataRequired

class InputForm(FlaskForm):
    Pclass = IntegerField(label='Enter Pclass',validators=[DataRequired()])
    Sex = SelectField(label='Sex',choices=['male','female'],validators=[DataRequired()])
    Age = FloatField(label='Age',validators=[DataRequired()])
    Fare = FloatField(label='Ticket Price',validators=[DataRequired()])
    Embarked = SelectField(label='Port/Embarked',choices=['S','C','Q'],validators=[DataRequired()])
    Family_Members = IntegerField(label='Number of Family Members',validators=[DataRequired()])
    Submit = SubmitField('Predict')