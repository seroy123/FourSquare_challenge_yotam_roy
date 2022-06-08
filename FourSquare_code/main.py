import os.path
import pandas as pd
import numpy as np
from os import getcwd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import extract_features
from NearestNeighborClassify import NearestNeighborClassify
from extract_features import ExtractFeatures


def get_data(path):
    test_data = pd.read_csv(path+r'\test.csv')
    train_data = pd.read_csv(path + r'\train.csv')
    return train_data, test_data


def get_labels(train_data, path, get_csv=False):
    if get_csv:
        labels_df = pd.DataFrame()
        for index, value in enumerate(train_data["id"].iloc[0:19]):
            to_append = train_data["id"].iloc[index]
            for poi_ind,POI in enumerate(train_data["point_of_interest"]):
                if POI == train_data["point_of_interest"].iloc[index] and index != poi_ind:
                    to_append += ' ' + train_data["id"].iloc[poi_ind]
            labels_df[train_data["id"].iloc[index]] = pd.Series(to_append)
        labels_df.T.to_csv(path + r'\labels.csv', index=True)
    return train_data["point_of_interest"]


def get_features(data):
    latitude_features = ExtractFeatures.latitude(data)
    longitude_features = ExtractFeatures.longitude(data)
    name_features = ExtractFeatures.name(data)
    features = np.concatenate((
        latitude_features[np.newaxis],
        longitude_features[np.newaxis],
        name_features[np.newaxis]
    ),axis=0)
    return features.T


def cross_validate(features, labels, k_fold : int, classifier):
    return cross_val_score(classifier, features, labels, cv=k_fold)


def get_pred(model, test_features):
    return model.predict(test_features)


def save_model_for_evaluation(test_data, test_predicion, path):
    df_to_save = pd.DataFrame(test_data["PassengerId"])
    df_to_save['Survived'] = list(test_predicion)
    df_to_save.to_csv(path+r'\solution.csv',index=False)


if __name__ == "__main__":
    # metric hyper-parameters: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’,
    # ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’][‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’,
    # ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    # [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]
    classifier = NearestNeighbors(n_neighbors=20,algorithm="brute", metric="chebyshev")
    path = os.path.dirname(os.path.dirname(getcwd()))+r'\data'
    train_data, test_data = get_data(path)
    train_labels = get_labels(train_data, path)
    train_features = get_features(train_data)
    get_classification = NearestNeighborClassify(classifier, id_list=train_data["id"], path=path)
    get_classification.fit(train_features=train_features, labels=train_labels)
    prediction_index = get_classification.get_duplicates(test_features=train_features[0:19, :], delta=0.5)
    get_classification.predict_proba(prediction_index)
