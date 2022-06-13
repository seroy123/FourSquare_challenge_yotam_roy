import numpy as np
import pandas as pd
import textdistance

class NearestNeighborClassify:
    def __init__(self, classifier, train_id_list, test_id_list, path,alpha,delta,distance):
        #:param delta: hyper-parameter. similarity method. too high will result false alarm. too low will result in misses
        #:param alpha: hyper-parameter. weight on distance from name category
        self.classifier = classifier
        self.features = None
        self.labels = None
        self.name_distance_vec = None
        self.name_features = None
        self.id_list = train_id_list
        self.path = path
        self.test_id_list = test_id_list
        self.alpha = alpha
        self.delta = delta
        self.distance = distance

    def name_distance(self,train_indeces, test_name):
        self.name_distance_vec = []
        name_features_train = self.name_features[train_indeces]
        for train_name in name_features_train:
            temp_distance = self.match_param_to_distance(test_name.replace(" ", ""), train_name.replace(" ", ""))
            self.name_distance_vec.append(temp_distance)

    def fit(self, train_features, labels):
        self.features = train_features[:, 0:2]
        self.name_features = train_features[:, 2]
        self.labels = labels
        self.classifier.fit(self.features)

    def predict(self, test_features):
        """
        This function receives the test features and a delta. it returns the prediction in a format that
        mathes the submission sample
        :param test_features: test_features
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
                                                         self.name_distance_vec[distance] * self.alpha
                if self.test_id_list[sample] == self.id_list[raw_prediction[sample][1][0][distance]]:
                    raw_prediction[sample][0][0][distance] = 999999  # make sure it wont get into the next if
                if distance == 0:
                    index_of_similar_POI.insert(0,sample)  # manually put himself(business) first in the list
                if raw_prediction[sample][0][0][distance] < self.delta:
                    index_of_similar_POI.append(raw_prediction[sample][1][0][distance])
            index_of_sample_and_similar_POI += [index_of_similar_POI]
        # Get prediction in a df that contains id's that matches the format of sample_submission
        for ind, pred in enumerate(index_of_sample_and_similar_POI):
            if pred:
                prediction_df[self.id_list.iloc[pred[0]]] = pd.Series(' '.join(list(self.id_list.iloc[pred])))
            else:
                prediction_df[self.test_id_list[ind]] = pd.Series(self.test_id_list[ind])
        prediction_by_format = prediction_df.T
        prediction_by_format['id'] = prediction_by_format.index
        prediction_by_format.columns = ['matches', 'id']
        prediction_by_format = pd.concat((prediction_by_format['id'], prediction_by_format['matches']), axis=1)
        return prediction_by_format

    def predict_proba(self, prediction):
        """
        This function receives the prediction and loads that labels to calculate the accuracy of the model
        :param prediction:
        :return:
        """
        labels = pd.read_csv(self.path + r'\labels.csv')
        labels = labels.iloc[0:700]
        counter = 0
        hit = 0
        miss = 0
        for sample in range(len(labels)):
            for guess in range(len(prediction['matches'].iloc[sample].split(' '))):
                if prediction['matches'].iloc[sample].split(' ')[guess] in labels.iloc[sample][1].split(' '):
                    hit += 1
                else:
                    miss += 1
                counter += 1
            if len(prediction['matches'].iloc[sample].split(' ')) < len(labels.iloc[sample][1].split(' ')):
                # punish me only if i guess too little. if i guessed too much, i am already punished by miss
                miss += (len(labels.iloc[sample][1].split(' ')) - len(prediction['matches'].iloc[sample].split(' ')))
                counter += (len(labels.iloc[sample][1].split(' ')) - len(prediction['matches'].iloc[sample].split(' ')))
        print(f"predicion rate is {(hit / counter) * 100}%")
        return (hit/counter)*100

    def match_param_to_distance(self, text1, text2):
        distance = getattr(textdistance, self.distance).normalized_distance(text1, text2)
        return distance
