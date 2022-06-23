import numpy as np
import pandas as pd
import textdistance
from sklearn.preprocessing import minmax_scale
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from py_stringmatching import similarity_measure
from py_stringmatching import JaroWinkler
from py_stringmatching import OverlapCoefficient
from py_stringmatching import Affine
from py_stringmatching import NeedlemanWunsch


class NearestNeighborClassify:
    def __init__(self, classifier, train_id_list, test_id_list, path,alpha,delta,distance):
        # delta: hyper-parameter. similarity method. too high will result false alarm. too low will result in misses
        # alpha: hyper-parameter. weight on distance from name category
        self.classifier = classifier
        self.features = None
        self.labels = None
        self.name_features = None
        self.id_list = train_id_list
        self.path = path
        self.test_id_list = test_id_list
        self.alpha = alpha
        self.delta = delta
        self.distance = distance

    def name_distance_func(self,train_indeces, test_name):
        distance_vec = []
        name_features_train = self.name_features[train_indeces]
        for train_name in name_features_train:
            temp_distance = self.match_param_to_distance(test_name, train_name, 'name')
            distance_vec.append(temp_distance)
        return distance_vec

    def distance_func(self, train_indeces, test_feature,feature_ind, feature_type):
        distance_vec = []
        features_train = self.features[train_indeces,feature_ind]
        for train_name in features_train:
            temp_distance = self.match_param_to_distance(test_feature, train_name,feature_type)
            distance_vec.append(temp_distance)
        return distance_vec

    def fit(self, train_features, labels):
        sim_func = JaroWinkler().get_raw_score
        self.geo_features = train_features[:, 0:2]
        self.features = train_features
        self.name_features = train_features[:, -1]
        self.labels = labels
        self.classifier.fit(self.geo_features)
        self.diff_obj_name = similarity_measure.soft_tfidf.SoftTfIdf([self.features[:,-1]],sim_func=sim_func)
        self.diff_obj_address = similarity_measure.soft_tfidf.SoftTfIdf([self.features[:,3]],sim_func=sim_func)
        self.diff_obj_category =similarity_measure.soft_tfidf.SoftTfIdf([self.features[:,5]],sim_func=sim_func)
        self.diff_obj_other = similarity_measure.soft_tfidf.SoftTfIdf(sim_func=sim_func)

    def predict(self, test_features):
        """
        This function receives the test features and a delta. it returns the prediction in a format that
        mathes the submission sample
        :param test_features: geo_features
        :return:
        """
        prediction_df = pd.DataFrame()
        index_of_sample_and_similar_POI = []
        test_names = test_features[:, -1]
        geo_features = test_features[:, 0:2]
        raw_prediction = [self.classifier.kneighbors(POI[np.newaxis]) for POI in geo_features]
        # testing
        distance_vec = np.reshape(minmax_scale(np.ravel([i[0] for i in raw_prediction]),feature_range=(0,1000)),
                                  (len(test_names), 20))
        for i in range(len(raw_prediction)):
            raw_prediction[i][0][0] = distance_vec[i,:]
        # update name distance
        for sample in range(len(raw_prediction)):
            # Get Distances
            name_distance = self.name_distance_func(raw_prediction[sample][1][0], test_names[sample])
            state_distance = self.distance_func(raw_prediction[sample][1][0], test_features[sample, 2],2, 'other')
            address_distance = self.distance_func(raw_prediction[sample][1][0], test_features[sample, 3],3, 'address')
            country_distance = self.distance_func(raw_prediction[sample][1][0], test_features[sample, 4],4, 'other')
            category_distance = self.distance_func(raw_prediction[sample][1][0], test_features[sample, 5],5, 'category')

            index_of_similar_POI = []
            for distance in range(len(raw_prediction[sample][0][0])):
                raw_prediction[sample][0][0][distance] = (raw_prediction[sample][0][0][distance]* self.alpha[5] +
                                                         name_distance[distance] * self.alpha[0] +
                  address_distance[distance]*self.alpha[2] + category_distance[distance]*self.alpha[4]+
                country_distance[distance]*self.alpha[3]+state_distance[distance]*self.alpha[1])
                if self.test_id_list[sample] == self.id_list[raw_prediction[sample][1][0][distance]]:
                    raw_prediction[sample][0][0][distance] = 999999  # make sure it wont get into the next if
                if distance == 0:
                    index_of_similar_POI.insert(0,self.test_id_list[sample])  # manually put himself(business) first in the list
                if raw_prediction[sample][0][0][distance] < self.delta:
                    index_of_similar_POI.append(self.id_list.iloc[raw_prediction[sample][1][0][distance]])
            index_of_sample_and_similar_POI += [index_of_similar_POI]
        # Get prediction in a df that contains id's that matches the format of sample_submission
        for ind, pred in enumerate(index_of_sample_and_similar_POI):
            prediction_df[str(pred[0])] = pd.Series(' '.join(pred))
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
        labels = labels.iloc[0:3200]
        counter = 0
        hit = 0
        false_alarm = 0
        for sample in range(len(labels)):
            pred_list = prediction['matches'].iloc[sample].split(' ')
            label_list = labels.iloc[sample][1].split(' ')
            # calculate Jaccard's index for estimation
            for guess in range(len(pred_list)):
                if pred_list[guess] in label_list:
                    hit += 1
                else:
                    false_alarm += 1
                counter += 1
            if len(pred_list) < len(label_list):  # punish for misses
                counter += len(label_list) - len(pred_list)
        print(f"predicion rate is {(hit / counter) * 100}%")
        print(f"false alarm rate out of total errors is {(false_alarm/(counter-hit)) * 100}%")
        return (hit/counter)*100, (false_alarm/(counter-hit)) * 100

    def match_param_to_distance(self, text1, text2, feature):
        if self.distance == 'soft':
            if feature == 'name':
                distance = 1 - self.diff_obj_name.get_raw_score(list(text1), list(text2))
            elif feature == 'category':
                distance = 1 - self.diff_obj_category.get_raw_score(list(text1), list(text2))
            elif feature == 'address':
                distance = 1 - self.diff_obj_address.get_raw_score(list(text1), list(text2))
            else:
                distance = 1 - self.diff_obj_other.get_raw_score(list(text1), list(text2))
        else:
            distance = getattr(textdistance, self.distance).normalized_distance(text1, text2)

        return distance
