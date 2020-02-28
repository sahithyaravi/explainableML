from flask_login import current_user
from app.models import *
import pandas as pd
from app.config import *
import logging
from app.models import Labels


def write_to_db(df, dataset):
    logging.info(f"writing user labels to db {current_user.username}")
    for index, row in df.iterrows():
        label = Labels(dataset=dataset,
                       user_id=current_user.id,
                       row_id=index,
                       batch=row["cluster_id"],
                       label=row["label"])
        db.session.add(label)
        db.session.commit()


def fetch_all_unlabelled_data(dataset):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    df = pd.read_sql_table(f"{dataset}_cluster", database_url)
    return df


def get_labelled_indices(dataset, user):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    user_id = user
    dataset_name = dataset
    sql = f"SELECT row_id from label where user_id={user_id} and dataset='{dataset_name}'"
    df = pd.read_sql_query(sql=sql, con=database_url)
    query_indices = df['row_id'].values
    return query_indices


def clear_labels():
    db.session.query(Labels).delete()
    db.session.commit()


def get_labelled_data(dataset_name):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    sql = f"SELECT * from label where dataset='{dataset_name}'"
    df = pd.read_sql_query(sql=sql, con=database_url)
    return df



