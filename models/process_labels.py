from app.utils import get_labelled_data, fetch_all_unlabelled_data
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


class Retrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None
        self.df_pool = None
        self.df_train = None

    def process_labels_from_all_users(self):
        labels = get_labelled_data(self.dataset)
        grouped_labels = labels.groupby(['row_id'])
        # if labels['user_id'].nunique() > 2:
        mean_labels = grouped_labels["label"].agg(pd.Series.mode)
        mean_labels = mean_labels.to_frame()
        df = fetch_all_unlabelled_data('davidson_dataset')
        labels.sort_values(by='row_id')
        df["labels"] = mean_labels['label']
        self.df_pool = df[df["labels"] == -1]
        self.df_train = df[df["labels"] != -1]
        self.df = df


r = Retrainer('davidson_dataset')
r.process_labels_from_all_users()

