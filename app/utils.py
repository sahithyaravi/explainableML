from flask_login import current_user
from app.models import *
import pandas as pd
from app.config import *
import logging
from app.models import Labels, Timer


def write_to_db(df, dataset):
    logging.info(f"writing user labels to db {current_user.username}")
    for index, row in df.iterrows():
        label = Labels(dataset=dataset,
                       user_id=current_user.id,
                       row_id=index,
                       batch=row["cluster_id"],
                       label=row["label"],
                       round=row["round"])
        db.session.add(label)
        db.session.commit()


def time_to_db(user, time, dataset):
    logging.info(f"writing user labels to db {current_user.username}")

    label = Timer(dataset=dataset,
                  user_id=user,
                  time=time)

    db.session.add(label)
    db.session.commit()


def fetch_all_unlabelled_data(dataset):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    df = pd.read_sql_table(dataset, database_url)
    return df


def fetch_train(dataset):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    df = pd.read_sql_table(f"{dataset}_train", database_url)
    return df


def fetch_test(dataset):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    df = pd.read_sql_table(f"{dataset}_test", database_url)
    return df


def get_labelled_indices(dataset, user, rnd):
    database_url = Config.SQLALCHEMY_DATABASE_URI
    user_id = user
    dataset_name = dataset
    sql = f"SELECT row_id from label where user_id={user_id} and dataset='{dataset_name}' and round = {rnd}"
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


def get_labelled_indices_pkl(dataset, user, rnd):
    labelled_df = pd.read_pickle(f"{current_user.id}_{dataset}_user.pkl")
    if not labelled_df.empty:
        query_indices = labelled_df['row_id'].values
    else:
        query_indices = []
    return query_indices


def write_to_pkl(df, dataset):
    logging.info(f"writing user labels to pkl {current_user.username}")
    new_df = pd.DataFrame()
    new_df["dataset"] = dataset
    new_df["user_id"] = current_user.id
    new_df["row_id"] = df["index"].values
    new_df["batch"] = df["cluster_id"].values
    new_df["label"] = df["label"].values
    new_df["round"] = df["round"].values
    labelled_df = pd.read_pickle(f"{current_user.id}_{dataset}_user.pkl")
    bigdata = labelled_df.append(new_df, ignore_index=True)
    bigdata.to_pickle(f"{current_user.id}_{dataset}_user.pkl")


def write_to_db_pkl(df, dataset, user_id):
    logging.info(f"writing user labels to db {user_id}")
    for index, row in df.iterrows():
        label = Labels(dataset=dataset,
                       user_id=user_id,
                       row_id=index,
                       batch=row["batch"],
                       label=row["label"],
                       round=row["round"])
        db.session.add(label)
        db.session.commit()


def get_all_user_labels():
    database_url = Config.SQLALCHEMY_DATABASE_URI
    sql = f"SELECT * from label"
    df = pd.read_sql_query(sql=sql, con=database_url)
    df.to_pickle('all_labels.pkl')
    return df

# import os
# path ="../../"
# files = os.listdir(path)
# print(files)
# files_txt = [i for i in files if i.endswith('user.pkl')]
# labelled_df = pd.read_pickle(f"{current_user.id}_{dataset}.pkl")
# write_to_db_pkl(labelled_df, dataset=dataset)
# os.remove(f"{current_user.id}_{dataset}.pkl")