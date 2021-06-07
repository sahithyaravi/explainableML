from app.utils import *
import plotly.express as px
dataset = 'davidson_dataset_cluster'

df_labels = get_all_user_labels()
df_labels = df_labels[df_labels["dataset"] == dataset]
df_truth = fetch_all_unlabelled_data(dataset)
time_elapsed = fetch_all_unlabelled_data('time_elapsed')


users = df_labels["user_id"].unique()

user_stats = {}
for user_id in users:
    labels_user = df_labels[df_labels["user_id"] == user_id]
    print(df_truth.shape, labels_user.shape)
    correct_labels = (labels_user['label'].isin(df_truth['truth']).value_counts().to_dict())
    user_stats[user_id] = correct_labels


stats_df = pd.dataframe.from_dict(user_stats)
print(stats_df.head())
# px.bar(user_stats)

