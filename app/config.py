import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """
    The location of apps database
    """
    # read this from JSON
    PASSWORD = '1993sahi11'
    database_url = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
    SQLALCHEMY_DATABASE_URI = database_url or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'abcd'

