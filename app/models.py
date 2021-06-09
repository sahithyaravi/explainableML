from flask_login import UserMixin
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
import os, sys
from app import db, login


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    name = db.Column(db.String(64), index=True, unique=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(128))
    english = db.Column(db.String(128))
    ml_experience = db.Column(db.String(128))
    password_hash = db.Column(db.String(128))
    email = db.Column(db.String(64), index=True, unique=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return '<User {}>'.format(self.username)


class Labels(db.Model):
    __tablename__ = 'label'
    id = db.Column(db.Integer, primary_key=True)
    dataset = db.Column(db.String(64), index=True)
    user_id = db.Column(db.Integer, index=True)
    row_id = db.Column(db.Integer, index=True)
    batch = db.Column(db.Integer, index=True)
    label = db.Column(db.Integer)
    round = db.Column(db.Integer, index=True)


class Timer(db.Model):
    __tablename__ = 'time_elapsed'
    id = db.Column(db.Integer, primary_key=True)
    dataset = db.Column(db.String(64), index=True)
    user_id = db.Column(db.Integer, index=True)
    time = db.Column(db.Float, index=True)
