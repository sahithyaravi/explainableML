import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    """
    The location of apps database
    """
    # PYTHON ANYWHERE
    PASSWORD = '1993sahi11'
    # SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
    #     username="sahithya",
    #     password="1993sahi11",
    #     hostname="sahithya.mysql.pythonanywhere-services.com",
    #     databasename="sahithya$shapely")
    #
    # SQLALCHEMY_POOL_RECYCLE = 299
    # SQLALCHEMY_POOL_SIZE = 3

    # LOCAL PC:
    DATABASE_URL = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or \
                              'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = 'abcd'

