import copy
import json
import os.path
import pandas as pd
import numpy as np
from os import getcwd
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from NearestNeighborClassify import NearestNeighborClassify
from extract_features import ExtractFeatures
from tqdm import tqdm
import random

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
            if index > 700 and index % 100 == 0:
                labels_df.T.to_csv(path + r'\labels.csv', index=True)
    return train_data["point_of_interest"]


def get_features(data):
    latitude_features = ExtractFeatures.latitude(data)
    longitude_features = ExtractFeatures.longitude(data)
    name_features = ExtractFeatures.name(data)
    country_features = ExtractFeatures.country(data)
    address_features = ExtractFeatures.address(data)
    state_features = ExtractFeatures.state(data)
    categories_features = ExtractFeatures.categories(data)

    features = np.concatenate((
        latitude_features[np.newaxis],
        longitude_features[np.newaxis],
        country_features[np.newaxis],
        address_features[np.newaxis],
        state_features[np.newaxis],
        categories_features[np.newaxis],
        name_features[np.newaxis],
    ),axis=0)
    return features.T

def get_pred(model, test_features):
    return model.predict(test_features)

def run_pipeline(delta,alpha, word_distance,geo_distance):
    classifier = NearestNeighbors(n_neighbors=20,metric=geo_distance)
    path = os.path.dirname(os.path.dirname(getcwd()))+r'\data'
    train_data, test_data = get_data(path)
    train_labels = get_labels(train_data, path)
    train_features = get_features(train_data)
    test_features = get_features(test_data)
    get_classification = NearestNeighborClassify(classifier, train_id_list=train_data["id"],
                                                 test_id_list=train_data["id"].iloc[0:3200], path=path,
                                                 delta=delta, alpha=alpha, distance=word_distance)
    get_classification.fit(train_features=train_features, labels=train_labels)
    prediction = get_classification.predict(test_features=train_features[0:3200])
    accuracy, FA = get_classification.predict_proba(prediction)
    # # for submission
    # get_classification = NearestNeighborClassify(classifier, train_id_list=train_data["id"],
    #                                              test_id_list=test_data["id"], path=path,
    #                                              delta=delta, alpha=alpha,distance=word_distance)
    # get_classification.fit(train_features=train_features, labels=train_labels)
    # prediction = get_classification.predict(test_features=test_features)
    return accuracy, FA, prediction


if __name__ == "__main__":
    ## grid search
    param_grid = {'delta': [1.25], 'alpha_name': [1],'alpha_state': [0],
                  'alpha_address': [0.4] ,'alpha_country': [0],'alpha_category': [0.6],
                  'alpha_distance': [0.1],
                  'distance':
        ['soft'], # 'levenshtein','jaro_winkler','jaro','damerau_levenshtein','jaccard'
        'geo_distance':['l1'] # ['cosine', 'euclidean', 'l1', 'l2', 'manhattan','chebyshev', 'correlation', 'dice',
                         #'hamming','jaccard', 'kulsinski', 'minkowski']
                  }
    grid = list(ParameterGrid(param_grid))
    random.shuffle(grid)
    max_pred = 0
    json_object = {}
    pred_stack = []
    for param in tqdm(grid, desc='Progress bar'):
        temp_alpha = [param['alpha_name'],param['alpha_state'],param['alpha_address'],param['alpha_country'],param['alpha_category'],param['alpha_distance']]
        temp_delta = param['delta']
        temp_distance = param['distance']
        temp_geo_distance = param['geo_distance']
        curr_pred, FA, prediction = run_pipeline(delta=temp_delta, alpha=temp_alpha, word_distance=temp_distance,geo_distance=temp_geo_distance)
        if len(pred_stack) < 10:
            pred_stack.insert(0,[copy.copy(curr_pred), copy.copy(FA), param])
            pred_stack = sorted(pred_stack, key=lambda a: a[0]-a[1])
        if (curr_pred - FA) > (pred_stack[0][0] - pred_stack[0][1]):
            del(pred_stack[0])
            pred_stack.insert(0, [copy.copy(curr_pred), copy.copy(FA), param])
            pred_stack = sorted(pred_stack, key=lambda a: a[0] - a[1])
            with open('json_data.json', 'w') as outfile:
                outfile.write(f'{pred_stack}')
            print('iteration results:')
            print(param)
            print(curr_pred)
    # metric hyper-parameters: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’,
    # ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’][‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’,
    # ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    # [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’]


    # for submission
    prediction.to_csv('submission.csv', index=False)
