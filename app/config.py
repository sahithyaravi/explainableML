import os
basedir = os.path.abspath(os.path.dirname(__file__))

from .run_config import *


class Config(object):
    """
    The location of apps database
    """
    # PYTHON ANYWHERE
    if SETUP == "web":
        SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
            username="sahithya",
            password=PASSWORD,
            hostname="sahithya.mysql.pythonanywhere-services.com",
            databasename="sahithya$shapely")

        SQLALCHEMY_POOL_RECYCLE = 299
        SQLALCHEMY_POOL_SIZE = 3
    else:
        # LOCAL PC:
        DATABASE_URL = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
        SQLALCHEMY_DATABASE_URI = DATABASE_URL or \
                                  'sqlite:///' + os.path.join(basedir, 'app.db')
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        SECRET_KEY = 'abcd'

