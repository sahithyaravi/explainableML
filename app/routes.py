from app import app, db
from flask import render_template, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from app.forms import LoginForm, RegisterForm
from app.models import User
from werkzeug.urls import url_parse


@app.route('/')
@app.route('/index/')
def index():
    return render_template('index.html', title='Home')


@app.route('/login/', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        next_page = url_for('index')
        return redirect(next_page)
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            error = 'Invalid username or password'
            return render_template('login.html', form=form, error=error)

        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)

    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout/')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register/', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    form = RegisterForm()
    print(form.username.data)
    if form.validate_on_submit():
        user = User(username=form.username.data,
                    name=form.name.data,
                    email=form.email.data,
                    gender=form.gender.data,
                    age=form.age.data,
                    english=form.english.data,
                    ml_experience = form.ml.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html', title='Register', form=form)
