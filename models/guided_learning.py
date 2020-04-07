import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cosine
from sqlalchemy import create_engine


class GuidedLearner:
    def __init__(self, df_train, df_test, df_pool, dataset, round):
        self.round = round
        self.df_train = df_train
        self.df_test = df_test
        self.df_pool = df_pool
        self.dataset = dataset
        self.tfid = TfidfVectorizer(max_features=5000)
        self.tfid.fit(self.df_train['text'].values)
        self.x_train = self.tfid.transform(self.df_train['text'].values).toarray()
        self.y_train = self.df_train['label'].values
        self.x_test = self.tfid.transform(self.df_test['text'].values).toarray()
        self.y_test = self.df_test['label'].values
        self.x_pool = self.tfid.transform(self.df_pool['text'].values).toarray()
        self.y_pool = self.df_pool['label'].values
        self.model = None
        self.shap_values_train = None
        self. shap_values_pool = None
        self.key_words_pos = None
        self.key_words_neg = None
        self.key_words = None
        PASSWORD = '1993sahi11'
        database_url = f"mysql+pymysql://root:{PASSWORD}@localhost/shapely"
        SQLALCHEMY_DATABASE_URI = database_url
        self.engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)

        self.df_train.to_sql(f"{self.dataset}_train", con=self.engine, if_exists="replace",
                             index=False)
        self.df_test.to_sql(f"{self.dataset}_test", con=self.engine, if_exists="replace",
                            index=False)

    def fit_svc(self, max_iter, C, kernel):
        self.model = SVC(max_iter=max_iter, C=C, kernel=kernel)
        self.model.fit(self.x_train, self.y_train)
        self.model.score(self.x_train, self.y_train)

    def get_shap_values(self):
        explainer = shap.LinearExplainer(self.model, self.x_train, feature_dependence="independent")
        # TODO extract feature importance value of each feature
        self.shap_values_train = explainer.shap_values(self.x_train)
        self.shap_values_pool = explainer.shap_values(self.x_pool)
        feature_names = np.array(self.tfid.get_feature_names())  # len(feature_names) = #cols in shap_values_pool
        shap.summary_plot(self.shap_values_pool, self.x_pool, feature_names=feature_names)

    def get_keywords(self):
        feature_names = np.array(self.tfid.get_feature_names())  # len(feature_names) = #cols in shap_values_pool
        arr = self.shap_values_pool.copy()
        arr[arr == 0] = np.nan
        arr_pos = self.shap_values_pool.copy()
        arr_neg = self.shap_values_pool.copy()
        arr_pos[arr_pos <= 0] = np.nan
        arr_neg[arr_neg >= 0] = np.nan
        indices = np.nanargmax(arr, axis=1)
        pos_indices = np.nanargmax(arr_pos, axis=1)
        neg_indices = np.nanargmin(arr_neg, axis=1)
        self.key_words = np.array([feature_names[indices]]).T
        self.key_words_pos = np.array([feature_names[pos_indices]]).T
        self.key_words_neg = np.array([feature_names[neg_indices]]).T

    def cluster_data_pool(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, max_iter=600)
        kmeans.fit(self.shap_values_pool)
        similarity_to_center = []
        for i, instance in enumerate(self.shap_values_pool):
            cluster_label = kmeans.labels_[i]  # cluster of this instance
            centroid = kmeans.cluster_centers_[cluster_label]  # cluster center of the cluster of that instance
            similarity = 1 - cosine(instance, centroid)  # 1- cosine distance gives similarity
            similarity_to_center.append(similarity)
        centroid_match = [None] * n_clusters
        centroid_indices = [None] * n_clusters
        for i, instance in enumerate(self.shap_values_pool):
            cluster_label = kmeans.labels_[i]
            if centroid_match[cluster_label] is None or similarity_to_center[i] > centroid_match[cluster_label]:
                centroid_indices[cluster_label] = i
                centroid_match[cluster_label] = similarity_to_center[i]
        pca = PCA(n_components=2)
       # principals = pca.fit_transform(self.shap_values_pool)
        tsne = TSNE(n_components=2, perplexity=20)
        principals = tsne.fit_transform(self.shap_values_pool)

        data = []
        collect = dict()
        color = ['hsl(' + str(h) + ',80%' + ',50%)' for h in np.linspace(0, 255, n_clusters)]
        df_final_labels = pd.DataFrame()
        for cluster_id in np.unique(kmeans.labels_):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)
            cluster_text = self.df_pool['text'].values[cluster_indices]

            cluster_truth = self.df_pool['label'].values[cluster_indices]
            center_index = centroid_indices[cluster_id]
            center_text = self.df_pool['text'].values[center_index]
            df_cluster = pd.DataFrame({'text': cluster_text})
            df_cluster['cluster_id'] = cluster_id
            df_cluster['centroid'] = False
            df_cluster['positive'] = self.key_words_pos[cluster_indices]
            df_cluster['negative'] = self.key_words_neg[cluster_indices]
            df_cluster['keywords'] = self.key_words[cluster_indices]
            df_cluster['truth'] = cluster_truth
            df_cluster = df_cluster.append({'text': center_text, 'cluster_id': cluster_id,
                                            'centroid': True}, ignore_index=True)
            df_final_labels = pd.concat([df_final_labels, df_cluster], ignore_index=True)

            cp = principals[cluster_indices]
            data.append(go.Scatter(x=cp[:, 0],
                                   y=cp[:, 1],
                                   mode='markers',
                                   hovertext=cluster_text,
                                   marker=dict(color=color[cluster_id],
                                               size=10),
                                   name='cluster ' + str(cluster_id)
                                   ))
            data.append(go.Scatter(x=[principals[center_index, 0]],
                                   y=[principals[center_index, 1]],
                                   mode='markers',
                                   marker=dict(color=color[cluster_id],
                                               size=15,
                                               line=dict(color='black', width=5)),
                                   name='centroid cluster ' + str(cluster_id)
                                   ))
            collect[cluster_id] = self.df_pool['text'].values[cluster_indices]

        fig = go.Figure(data=data)
        fig.show()
        df_final_labels.reset_index(drop=True, inplace=True)
        df_final_labels["round"] = self.round
        df_final_labels.to_sql(f"{self.dataset}_cluster", con=self.engine, if_exists="replace")




