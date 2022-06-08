import numpy as np
from functools import lru_cache
import pandas as pd

class NearestNeighborClassify:
    def __init__(self, classifier, id_list, path):
        self.classifier = classifier
        self.features = None
        self.labels = None
        self.name_distance_vec = None
        self.name_features = None
        self.id_list = id_list
        self.path = path

    def name_distance(self,train_indeces, test_name):
        name_features_train = self.name_features[train_indeces]
        self.name_distance_vec = [lev_dist(test_name, train_name) for\
            train_name in name_features_train]

    def fit(self, train_features, labels):
        self.features = train_features[:, 0:2]
        self.name_features = train_features[:, 2]
        self.labels = labels
        self.classifier.fit(self.features)

    def get_duplicates(self, test_features, delta=0.5):
        """
        This function receives the test features and a delta. it returns the prediciton in a format that
        mathes the submission sample
        :param test_features: test_features
        :param delta: hyper-parameter. similarity method. too high will result false alarm. too low will result in misses
        :return:
        """
        prediction_df = pd.DataFrame()
        index_of_sample_and_similar_POI = []
        test_names = test_features[:, 2]
        test_features = test_features[:, 0:2]
        raw_prediction = [self.classifier.kneighbors(POI[np.newaxis]) for POI in test_features]
        # update name distance
        for sample in range(len(raw_prediction)):
            self.name_distance(raw_prediction[sample][1][0], test_names[sample])
            index_of_similar_POI = []
            for distance in range(len(raw_prediction[sample][0][0])):
                raw_prediction[sample][0][0][distance] = raw_prediction[sample][0][0][distance] + \
                                                         self.name_distance_vec[distance] * 0.1
                if raw_prediction[sample][0][0][distance] < delta:
                    index_of_similar_POI.append(raw_prediction[sample][1][0][distance])
            index_of_sample_and_similar_POI += [index_of_similar_POI]
        # Get prediction in a df that contains id's that matches the format of sample_submission
        for pred in index_of_sample_and_similar_POI:
            prediction_df[self.id_list.iloc[pred[0]]] = pd.Series(' '.join(list(self.id_list.iloc[pred])))
        return prediction_df.T

    def predict_proba(self, prediction):
        """
        This function receives the prediction and loads that labels to calculate the accuracy of the model
        :param prediction:
        :return:
        """
        labels = pd.read_csv(self.path + r'\labels.csv')
        counter = 0
        hit = 0
        miss = 0
        for sample in range(len(labels)):
            for guess in range(len(prediction.iloc[sample][0].split(' '))):
                if prediction.iloc[sample][0].split(' ')[guess] in labels.iloc[sample][1].split(' '):
                    hit += 1
                else:
                    miss += 1
                counter += 1
            if len(prediction.iloc[sample][0].split(' ')) < len(labels.iloc[sample][1].split(' ')):
                # punish me only if i guess too little. if i guessed too much, i am already punished by miss
                miss += (len(labels.iloc[sample][1].split(' ')) - len(prediction.iloc[sample][0].split(' ')))
                counter += (len(labels.iloc[sample][1].split(' ')) - len(prediction.iloc[sample][0].split(' ')))
        print(f"predicion rate is {(hit / counter) * 100}%")
        return (hit/counter)*100


def lev_dist(a, b):
    '''
    This function will calculate the levenshtein distance between two input
    strings a and b

    params:
        a (String) : The first string you want to compare
        b (String) : The second string you want to compare

    returns:
        This function will return the distnace between string a and b.

    example:
        a = 'stamp'
        b = 'stomp'
        lev_dist(a,b)
        >> 1.0
    '''

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)
