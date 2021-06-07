from app.utils import write_to_db_pkl
import pandas as pd
import os

""" After a user labels all the batches, the labels are stored locally in a pickle file
This pythons script runs as a cron job and uploads all the labels to the database and automatically deletes the pickle files.
When you run the app locally, run this manually instead.

"""
path = '../'
files = os.listdir(path)
print(files)
files = [i for i in files if i.endswith('user.pkl')]
for file in files:
	user_id = file.split("_")[0]
	labelled_df = pd.read_pickle(os.path.join(path, file))
	dataset = file.replace("_user", "")
	dataset = file.replace(".pkl", "")
	dataset = dataset.replace(f"{user_id}_", "")
	print(file)
	write_to_db_pkl(labelled_df, dataset=dataset, user_id = int(user_id))
	os.remove(os.path.join(path, file))