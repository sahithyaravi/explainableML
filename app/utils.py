from flask_login import current_user
from app.models import *


def write_to_db(df, dataset):
    print("writing user to db", current_user.username)
    for index, row in df.iterrows():
        label = Labels(dataset=dataset,
                       user_id=current_user.id,
                       row_id=index,
                       batch=row["cluster_id"],
                       label=row["label"])
        db.session.add(label)
        db.session.commit()
