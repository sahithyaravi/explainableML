from flask import Flask
from flask_login import LoginManager
from app.config import Config
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from app.dashboard import register_dashapps
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
db.init_app(app)

login = LoginManager()
login.init_app(app)
login.login_view = 'login'
migrate.init_app(app, db)
register_dashapps(app)

from app import routes, models







