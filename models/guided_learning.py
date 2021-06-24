import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import plotly
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, homogeneity_score, v_measure_score, completeness_score
import seaborn as sns
import shap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean, cosine
from sqlalchemy import create_engine
from app.config import Config


class GuidedLearner:
    """ Contains helper functions to easily create data to be used by the guided annotation tool """
    def __init__(self, df_train, df_test, df_pool, df_individual, dataset, round=0):
        # print(df_individual.head())
        n_individual_rows = df_individual.shape[0]
        ids = []
        j = 0
        for i in range(0, n_individual_rows, 15):
            ids.extend([j]*15)
            j += 1
        ids = ids[:n_individual_rows]
        # print(len(ids), n_individual_rows)
        df_individual["cluster_id"] = ids
        self.round = round
        df_individual["round"] = self.round
        self.df_individual = df_individual
        self.df_train = df_train
        self.df_test = df_test
        self.df_pool = df_pool
        self.dataset = dataset
        self.model, self.tfid = None, None
        self.x_train, self.x_pool, self.x_test = None, None, None
        self.y_train, self.y_pool, self.y_test = None, None, None
        self.shap_values_train, self. shap_values_pool = None, None
        self.key_words_pos, self.key_words_neg, self.key_words = None, None, None

    def tfid_fit(self):
        self.tfid = TfidfVectorizer(max_features=5000)
        self.tfid.fit(self.df_train['processed'].values)
        self.x_train = self.tfid.transform(self.df_train['processed'].values).toarray()
        self.y_train = self.df_train['label'].values
        self.x_test = self.tfid.transform(self.df_test['processed'].values).toarray()
        self.y_test = self.df_test['label'].values
        self.x_pool = self.tfid.transform(self.df_pool['processed'].values).toarray()
        self.y_pool = self.df_pool['label'].values
        return self.tfid, self.x_train, self.x_test, self.x_pool, self.y_train, self.y_test, self.y_pool

    def grid_search_fit_svc(self, c=None):
        if c is None:
            c = [0.8, 1]
        max_iter = 1000
        best_f1 = 0
        model = None
        for c_option in c:
            m = SVC(max_iter=max_iter, C=c_option, kernel='linear', class_weight='balanced', probability=True)
            m.fit(self.x_train, self.y_train)
            predictions = m.predict(self.x_test)
            f1 = f1_score(predictions, self.y_test)
            if f1 > best_f1:
                self.model = m
                best_f1 = f1
        pred = self.model.predict(self.x_test)
        print("F1 score on test set ", f1_score(self.y_test, pred))
        print("Confusion matrix on test set ", confusion_matrix(self.y_test, pred))
        print("Accuracy test set", accuracy_score(self.y_test, pred))
        pred = self.model.predict(self.x_pool)
        print("F1 score on pool ", f1_score(self.y_pool, pred))
        print("Confusion matrix of final model on pool ", confusion_matrix(self.y_pool, pred))
        print("Accuracy of final model on pool", accuracy_score(self.y_pool, pred))
        explainer = shap.Explainer(self.model, self.x_train, feature_perturbation="independent")
        # TODO extract feature importance value of each feature
        self.shap_values_train = explainer.shap_values(self.x_train)
        self.shap_values_pool = explainer.shap_values(self.x_pool)
        feature_names = np.array(self.tfid.get_feature_names())  # len(feature_names) = #cols in shap_values_pool
        shap.summary_plot(self.shap_values_train, self.x_train, feature_names=feature_names)
        return self.model, explainer

    def fit_tree(self):
        estimators = [100, 500, 1000, 1500, 2000, 5000]
        features = ['auto', 0.8, 'sqrt']
        max_depth = [20, 30, 50, 100, None]
        best_model = None
        max_score = 0
        for e in estimators:
            for f in features:
                for m in max_depth:
                    self.model = GradientBoostingClassifier(n_estimators=e, max_features=f, max_depth=m)
                    self.model.fit(self.x_train, self.y_train)
                    # print("Score on train set", self.model.score(self.x_train, self.y_train))
                    pred = self.model.predict(self.x_test)
                    # print("Confusion matrix ", confusion_matrix(self.y_test, pred))
                    s = accuracy_score(self.y_test, pred)
                    # print("Accuracy on test set ", s)
                    if s > max_score:
                        max_score = s
                        best_model = self.model
        print("BEST MODEL", best_model.n_estimators, best_model.max_features, best_model.max_depth)
        self.model = best_model
        pred = self.model.predict(self.x_pool)
        print("Confusion matrix of best model on pool ", confusion_matrix(self.y_pool, pred))
        print("Accuracy of best model on pool", accuracy_score(self.y_pool, pred))

        explainer = shap.TreeExplainer(self.model, self.x_train)
        # TODO extract feature importance value of each feature
        self.shap_values_train = explainer.shap_values(self.x_train)
        self.shap_values_pool = explainer.shap_values(self.x_pool)
        feature_names = np.array(self.tfid.get_feature_names())  # len(feature_names) = #cols in shap_values_pool
        shap.summary_plot(self.shap_values_pool, self.x_pool, feature_names=feature_names)
        self.shap_values_pool = self.shap_values_pool
        return self.model, explainer

    def _get_keywords(self):
        print("shap values", len(self.shap_values_pool))
        feature_names = np.array(self.tfid.get_feature_names())  # len(feature_names) = #cols in shap_values_pool
        arr = self.shap_values_pool.copy()
        arr[arr == 0] = np.nan
        arr_pos = self.shap_values_pool.copy()
        arr_neg = self.shap_values_pool.copy()
        arr_pos[arr_pos <= 0] = np.nan
        arr_neg[arr_neg >= 0] = np.nan
        abs_arr = np.abs(arr)
        indices = np.nanargmax(abs_arr, axis=1)
        pos_indices = np.nanargmax(arr_pos, axis=1)
        neg_indices = np.nanargmin(arr_neg, axis=1)
        self.key_words = np.array([feature_names[indices]]).T
        self.key_words_pos = np.array([feature_names[pos_indices]]).T
        self.key_words_neg = np.array([feature_names[neg_indices]]).T

    def cluster_data_pool(self, uncertainty_visible=False,
                          pca=True, pca_components=100, cluster_sizes=None):
        if not cluster_sizes:
            cluster_sizes = 20
        # Dimensionality reduction
        pca = PCA(n_components=pca_components)
        principals = pca.fit_transform(self.shap_values_pool)
        tsne = TSNE(n_components=2, perplexity=20)
        principals_tsne = tsne.fit_transform(self.shap_values_pool)

        self._get_keywords()

        # Uncerainty
        colorscale = [[0, 'mediumturquoise'], [1, 'salmon']]
        classwise_uncertainty = self.model.predict_proba(self.x_pool)
        uncertainty = 1 - np.max(classwise_uncertainty, axis=1)

        # cluster shapely values
        print("Finding optimal cluster size")

        if pca:
            n_clusters = self.find_cluster_size(n_clusters_range=cluster_sizes, data= principals, labels=self.y_pool,
                                                n_iters=1)
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, max_iter=600)
            kmeans.fit(principals)
        else:
            n_clusters = self.find_cluster_size(n_clusters_range=cluster_sizes, data= self.shap_values_pool, labels=self.y_pool,
                                                n_iters=1)
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1, max_iter=600)
            kmeans.fit(self.shap_values_pool)

        print("Homogenity score", homogeneity_score(self.y_pool, kmeans.labels_))
        print("v measure score", v_measure_score(self.y_pool, kmeans.labels_))

        data = []
        collect = dict()
        color = ['hsl(' + str(h) + ',80%' + ',50%)' for h in np.linspace(0, 255, n_clusters)]
        df_final_labels = pd.DataFrame()
        for cluster_id in np.unique(kmeans.labels_):
            cluster_indices = np.where(kmeans.labels_ == cluster_id)
            cluster_text = self.df_pool['text'].values[cluster_indices]
            print("Cluster id", cluster_id, cluster_text.shape, np.unique(cluster_text).shape)
            cluster_truth = self.df_pool['label'].values[cluster_indices]
            # center_index = centroid_indices[cluster_id]
            # center_text = self.df_pool['text'].values[center_index]
            df_cluster = pd.DataFrame({'text': cluster_text})
            df_cluster['cluster_id'] = cluster_id
            # df_cluster['centroid'] = False
            df_cluster['positive'] = self.key_words_pos[cluster_indices]
            df_cluster['negative'] = self.key_words_neg[cluster_indices]
            df_cluster['keywords'] = self.key_words[cluster_indices]
            df_cluster['truth'] = cluster_truth
            # df_cluster = df_cluster.append({'cluster_id': cluster_id,
            #                                 'centroid': True}, ignore_index=True)
            df_final_labels = pd.concat([df_final_labels, df_cluster], ignore_index=True)

            cp = principals_tsne[cluster_indices]
            data.append(go.Scatter(x=cp[:, 0],
                                   y=cp[:, 1],
                                   mode='markers',
                                   hovertext=cluster_text,
                                   marker=dict(color=color[cluster_id],
                                               size=10),
                                   name='cluster ' + str(cluster_id)
                                   ))
            # data.append(go.Scatter(x=[principals[center_index, 0]],
            #                        y=[principals[center_index, 1]],
            #                        mode='markers',
            #                        marker=dict(color=color[cluster_id],
            #                                    size=15,
            #                                    line=dict(color='black', width=5)),
            #                        name='centroid cluster ' + str(cluster_id)
            #                        ))
            data.append(go.Heatmap(x=cp[:, 0],
                                   y=cp[:, 1],
                                   z=uncertainty[cluster_indices],
                                   name='uncertainity map',
                                   visible=uncertainty_visible,
                                   showscale=False,
                                   colorscale=colorscale,
                                   ))
            collect[cluster_id] = self.df_pool['text'].values[cluster_indices]

        fig = go.Figure(data=data)
        fig.show()
        return df_final_labels, uncertainty

    def save_to_db(self, df_final_labels):
        SQLALCHEMY_DATABASE_URI = Config.SQLALCHEMY_DATABASE_URI
        self.engine = create_engine(SQLALCHEMY_DATABASE_URI, echo=False)
        self.df_train.to_sql(f"{self.dataset}_train", con=self.engine, if_exists="replace",
                             index=False)
        self.df_test.to_sql(f"{self.dataset}_test", con=self.engine, if_exists="replace",
                            index=False)
        self.df_individual.to_sql(f"{self.dataset}_noshap", con=self.engine, if_exists="replace",
                             index=False
                             )
        df_final_labels.reset_index(drop=True, inplace=True)
        df_final_labels["round"] = self.round
        df_final_labels.to_sql(f"{self.dataset}_cluster", con=self.engine, if_exists="replace")

    def find_cluster_size(self, n_clusters_range, data, labels, n_iters=1):
        homogeneity_scores = []
        v_measure_scores = []
        completeness_scores = []
        ranges = n_clusters_range #list(range(10, 110, 10))
        for k in ranges:
            vavg = 0
            havg = 0
            cavg = 0
            for i in range(n_iters):
                kmeans = KMeans(n_clusters=k, n_jobs=-1)
                kmeans.fit(data)
                v = v_measure_score(labels_pred=kmeans.labels_, labels_true=labels)
                h = homogeneity_score(labels_pred=kmeans.labels_, labels_true=labels)
                c = completeness_score(labels_pred=kmeans.labels_, labels_true=labels)
                vavg += v
                havg += h
                cavg += c
            homogeneity_scores.append(havg / n_iters)
            v_measure_scores.append(vavg / n_iters)
            completeness_scores.append(cavg / n_iters)
            print(k, "done")

        data = [go.Scatter(x=ranges, y=homogeneity_scores, mode="lines", name="homogeneity"),
                go.Scatter(x=ranges, y=v_measure_scores, mode="lines", name="v_measure"),
                go.Scatter(x=ranges, y=completeness_scores, mode="lines", name="completeness")
                ]
        fig = go.Figure(data=data)
        fig.update_layout(xaxis_title="no of clusters")
        fig.show()
        max_homogeneity_index = v_measure_scores.index(max(v_measure_scores))
        return ranges[max_homogeneity_index]

    def find_similarity(self, kmeans, n_clusters):
        similarity_to_center = []
        # find centroid of cluster
        centroid = False
        if centroid:
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



