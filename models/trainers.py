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

    def train_test_pool_split(self, df, train_frac=0.6, test_frac=0.2, pool_frac=0.2, unguided='same'):
        """

        :param df: the dataset for train test split
        :param stratify: whether the split needs to be stratified or not
        :return: train set, test set, pool set(for guided learning), individual set(for individual labelling)
        """
        if unguided not in ["same", "different"]:
            return TypeError("Can only be same or different")

        df.dropna(inplace=True)
        df.drop_duplicates(subset=['processed'], inplace=True, keep='last')
        seed = 150
        np.random.seed(seed)
        stratify = False
        if stratify:
            self.df_train, df_valid = train_test_split(
                df, test_size=train_frac+pool_frac, shuffle=True, stratify=df.label,
                random_state=seed)
            self.df_test, df_rest = train_test_split(df_valid, test_size=0.5, shuffle=True, stratify=df_valid.label,
                                                random_state=seed)
            self.df_pool, self.df_individual = train_test_split(df_rest, test_size=0.5, shuffle=True, stratify=df_rest.label,
                                                      random_state=seed)
        else:
            indices = np.random.randint(low=0, high=df.shape[0], size=df.shape[0])
            train_indices = indices[0:round(train_frac * df.shape[0])]
            test_end = train_frac+test_frac
            pool_end = train_frac+test_frac+pool_frac
            test_indices = indices[round(train_frac * df.shape[0]): round(test_end * df.shape[0])]
            pool_indices = indices[round(test_end * df.shape[0]): round(pool_end * df.shape[0])]
            if unguided =="same":
                individual_indices = pool_indices
            else:
                individual_indices = indices[round(pool_end* df.shape[0]):]

            self.df_train = df.iloc[train_indices]
            self.df_test = df.iloc[test_indices]
            self.df_pool = df.iloc[pool_indices]
            print("pool", self.df_pool.shape)
            self.df_pool.drop_duplicates(subset=['processed'], inplace=True, keep='last')
            print(self.df_pool.shape)
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




