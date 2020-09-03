import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """
    The location of apps database
    """
    # read this from JSON
    PASSWORD = '1993sahi11'
    #DATABASE_URL = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"

    DATABASE_URL = f"mysql+pymysql://sahithya:{PASSWORD}@sahithya.mysql.pythonanywhere-services.com/sahithya$shapely" #PYTHON ANYWHERE
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'abcd'

