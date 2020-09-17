import numpy as np
import pandas as pd
from models.guided_learning import GuidedLearner
from models.retrainer import Retrainer


class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset

    def init_train(self):
        """Initial training of the dataset using guided learner"""
        df = pd.read_csv(f"datasets/{self.dataset}.csv")
        df = df.sample(100)
        df.columns = ['index','text', 'label', 'processed']
        print(df.shape)
        df.drop_duplicates(subset=['text'], inplace=True)
        print(df.shape)

        indices = np.random.randint(low=0, high=df.shape[0], size=df.shape[0])
        train_indices = indices[0:round(0.5 * df.shape[0])]
        test_indices = indices[round(0.5 * df.shape[0]): round(0.8 * df.shape[0])]
        pool_indices = indices[round(0.8 * df.shape[0]): round(0.9 * df.shape[0])]
        individual_indices = indices[round(0.9 * df.shape[0]):]

        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]
        df_pool = df.iloc[pool_indices]
        df_individual = df.iloc[individual_indices]
        df_individual.reset_index(inplace=True)
        df_individual.drop("index", inplace=True, axis=1)
        df_individual.reset_index(inplace=True)

        learner = GuidedLearner(df_train, df_test, df_pool, df_individual, self.dataset, 1)
        learner.fit_svc(max_iter=1000, C=5, kernel='linear')
        #learner.fit_MLP()
        #learner.fit_tree()
        # learner.get_keywords()
        # learner.cluster_data_pool(n_clusters=10)

    def retrain(self):
        r = Retrainer(self.dataset)
        r.process_labels_from_all_users()
        r.retrain_model()




