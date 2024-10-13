from wtforms import TextAreaField,SubmitField
from wtforms.validators import DataRequired
from flask_wtf import FlaskForm

class InputForm(FlaskForm):
    message = TextAreaField(label='Enter a message',validators=[DataRequired()])
    submit  = SubmitField(label='Predict')

