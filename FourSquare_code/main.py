import os.path
import pandas as pd
import numpy as np
from os import getcwd
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from NearestNeighborClassify import NearestNeighborClassify
from extract_features import ExtractFeatures
from tqdm import tqdm

def get_data(path):
    test_data = pd.read_csv(path+r'\test.csv')
    train_data = pd.read_csv(path + r'\train.csv')
    return train_data, test_data


def get_labels(train_data, path, get_csv=False):
    if get_csv:
        labels_df = pd.DataFrame()
        for index, value in tqdm(enumerate(train_data["id"].iloc[0:50000]), desc='Progress bar'):
            to_append = train_data["id"].iloc[index]
            for poi_ind,POI in enumerate(train_data["point_of_interest"]):
                if POI == train_data["point_of_interest"].iloc[index] and index != poi_ind:
                    to_append += ' ' + train_data["id"].iloc[poi_ind]
            labels_df[train_data["id"].iloc[index]] = pd.Series(to_append)
            if index > 200 and index % 100 == 0:
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

def run_pipeline(delta,alpha, distance):
    classifier = NearestNeighbors(n_neighbors=20,algorithm="brute")
    path = os.path.dirname(os.path.dirname(getcwd()))+r'\data'
    train_data, test_data = get_data(path)
    train_labels = get_labels(train_data, path)
    train_features = get_features(train_data)
    test_features = get_features(test_data)
    get_classification = NearestNeighborClassify(classifier, train_id_list=train_data["id"],
                                                 test_id_list=train_data["id"].iloc[0:700], path=path,
                                                 delta=delta, alpha=alpha,distance=distance)
    get_classification.fit(train_features=train_features, labels=train_labels)
    prediction = get_classification.predict(test_features=train_features[0:700])
    return get_classification.predict_proba(prediction)


if __name__ == "__main__":
    ## grid search
    param_grid = {'delta': np.linspace(0.01,0.1,20), 'alpha':[0.1],'distance':
        ['levenshtein']}#,'jaro_winkler','jaro','damerau_levenshtein','ratcliff_obershelp','jaccard' # np.linspace(0.08,0.1,2)
    grid = ParameterGrid(param_grid)
    max_pred = 0
    for param in tqdm(grid, desc='Progress bar'):
        temp_alpha = param['alpha']
        temp_delta = param['delta']
        temp_distance = param['distance']
        curr_pred = run_pipeline(delta=temp_delta, alpha=temp_alpha, distance=temp_distance)
        if curr_pred > max_pred:
            max_pred = curr_pred
            max_params = param
    # metric hyper-parameters: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’,
    # ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’][‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’,
    # ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    # [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]


    # # for submission
    # prediction_index['id'] = prediction_index.index
    # prediction_index.columns = ['matches', 'id']
    # prediction_index = pd.concat((prediction_index['id'], prediction_index['matches']), axis=1)
    # prediction_index.to_csv('submission.csv',index=False)
