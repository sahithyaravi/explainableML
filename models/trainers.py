import numpy as np
from models.retrainer import Retrainer
from sklearn.model_selection import train_test_split


class Trainer:
    def __init__(self, dataset_name):
        self.df_train = None
        self.df_pool = None
        self.df_test = None
        self.df_individual = None
        self.dataset_name = dataset_name

    def train_test_pool_split(self, df, stratify=False):
        """

        :param df: the dataset for train test split
        :param stratify: whether the split needs to be stratified or not
        :return: train set, test set, pool set(for guided learning), individual set(for individual labelling)
        """
        df.dropna(inplace=True)
        # df = df.sample(500)
        # df.columns = ['index','text', 'label', 'processed']
        print(df.shape)
        df.drop_duplicates(subset=['processed'], inplace=True, keep='last')
        df.drop_duplicates(subset=['processed'], inplace=True)
        print(df.shape)
        seed = 200
        np.random.seed(seed)
        if stratify:
            self.df_train, df_valid = train_test_split(
                df, test_size=0.4, shuffle=True, stratify=df.label,
                random_state=seed)
            self.df_test, df_rest = train_test_split(df_valid, test_size=0.75, shuffle=True, stratify=df_valid.label,
                                                random_state=seed)
            self.df_pool, self.df_individual = train_test_split(df_rest, test_size=0.3, shuffle=True, stratify=df_rest.label,
                                                      random_state=seed)
        else:
            indices = np.random.randint(low=0, high=df.shape[0], size=df.shape[0])
            train_indices = indices[0:round(0.8 * df.shape[0])]
            test_indices = indices[round(0.8 * df.shape[0]): round(0.9 * df.shape[0])]
            pool_indices = indices[round(0.9 * df.shape[0]): round(0.95 * df.shape[0])]
            individual_indices = indices[round(0.95 * df.shape[0]):]

            self.df_train = df.iloc[train_indices]
            self.df_test = df.iloc[test_indices]
            self.df_pool = df.iloc[pool_indices]
            self.df_individual = df.iloc[individual_indices]
            self.df_individual.reset_index(inplace=True)
            self.df_individual.drop("index", inplace=True, axis=1)
            self.df_individual.reset_index(inplace=True)
        return self.df_train, self.df_test, self.df_pool, self.df_individual

    #
    # def retrain(self):
    #     r = Retrainer(self)
    #     r.process_labels_from_all_users()
    #     r.retrain_model()




