from app.utils import get_labelled_data, fetch_all_unlabelled_data, fetch_train, fetch_test
import pandas as pd
from models.guided_learning import GuidedLearner
pd.set_option('display.expand_frame_repr', False)


class Retrainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.df = None
        self.df_pool = None
        self.df_train = None
        self.df_test = None
        self.round = 0

    def process_labels_from_all_users(self):
        labels = get_labelled_data(self.dataset)
        grouped_labels = labels.groupby(['row_id'])
        # if labels['user_id'].nunique() > 2:
        mean_labels = grouped_labels["label"].agg(pd.Series.mode)
        mean_labels = mean_labels.to_frame()
        print(mean_labels)
        df = fetch_all_unlabelled_data('davidson_dataset')  # Now we have labelled this data
        train = fetch_train(self.dataset)
        labels.sort_values(by='row_id')
        df["label"] = labels['label']
        self.round = df["round"][0] + 1

        self.df_pool = df.loc[df["label"] == -1]
        self.df_train = df.loc[df["label"] != -1]
        self.df_test = fetch_test(self.dataset)
        self.df_train = self.df_train.append(train, ignore_index=True)
        self.df = df
        print(self.df_train.columns)
        # self.df_train.drop('level_0', axis=1)
        print(train.shape, self.df_train.shape, self.df_pool.shape)

    # def retrain_model(self):
    #     learner = GuidedLearner(self.df_train, self.df_test, self.df_pool, self.dataset, self.round)
    #     learner.fit_svc(max_iter=2000, C=1, kernel='linear')
    #     learner.get_shap_values()
    #     learner.get_keywords()
    #     learner.cluster_data_pool(n_clusters=20)




