import pandas as pd
from models.trainers import Trainer
from app.utils import clear_labels
from models.guided_learning import GuidedLearner
pd.set_option('display.max_colwidth', 1000)

clear_labels()
dataset_name = 'gao_dataset'
df = pd.read_csv(f"datasets/{dataset_name}.csv")  # substitute other datasets in similar format
t = Trainer(dataset_name=dataset_name) # the name which you want for the tables in the database
df_train, df_test, df_pool, df_individual = t.train_test_pool_split(df, stratify=False,
                                                                    train_frac=0.6,
                                                                    test_frac=0.8,
                                                                    pool_frac=0.9
                                                                    )
df_pool.to_csv('pool.csv')


print("Number of points to be clustered: ", df_pool.shape)
print("Number of points to be indiviually labeled: ", df_individual.shape)

learner = GuidedLearner(df_train, df_test, df_pool, df_individual, dataset_name, 1)
tfid, x_train, x_test, x_pool, y_train, y_test, y_pool = learner.tfid_fit()

model, explainer = learner.grid_search_fit_svc(c=[0.8, 1])
df_final_labels, uncertainty = learner.cluster_data_pool(pca=True,pca_components=100, cluster_sizes=[10])
df_final_labels.to_csv('final_labels.csv')
# # learner.save_to_db(df_final_labels)
