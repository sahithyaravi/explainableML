from flask_wtf import FlaskForm
from wtforms import BooleanField, PasswordField, StringField, SubmitField, RadioField, SelectField, SelectMultipleField, TextAreaField
from wtforms.validators import DataRequired
from wtforms.fields.html5 import EmailField, IntegerField

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')
    remember_me = BooleanField('Remember Me')


class RegisterForm(FlaskForm):
    username = StringField('Choose username', validators=[DataRequired()])
    password = PasswordField('Set your password', validators=[DataRequired()])

    name = StringField('Name', validators=[DataRequired()])

    email = EmailField('Email', validators=[DataRequired()])

    age = IntegerField('Age', validators=[DataRequired()])

    gender = RadioField('Gender', choices=[
        ('male', 'Male'),
        ('female', 'Female'),
        ('neutral', 'Neutral'),
        ('undisclosed', 'No preference'),
    ], validators=[DataRequired()])

    english = RadioField('Which level of english fluency do you identify yourself with', choices=[
        ('beginner', 'beginner'),
        ('intermediete', 'intermediete'),
        ('Advanced', 'Advanced'),
    ], validators=[DataRequired()])

    ml = SelectField(
        'Which category best describes your experience with working on data science/machine learning?',
        choices=[
            # ('', 'Select your path'),
            ('enthusiast', 'No experience'),
            ('beginner', 'Beginner with 0-1 years of experience'),
            ('intermediete', 'Intermediete with 1-5 years of experience'),
            ('expert', 'Advanced with more than 5 years of experience'),
        ])
    submit = SubmitField('Register')
