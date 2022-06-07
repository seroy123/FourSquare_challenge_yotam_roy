import pandas as pd
import numpy as np
from os import getcwd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def get_data(path):
    test_data = pd.read_csv(path+r'\test.csv')
    train_data = pd.read_csv(path + r'\train.csv')
    train_labels = np.array(train_data["point_of_interest"])
    return train_data, train_labels, test_data

def get_features(data):
    age_features = np.array(data["Age"].fillna(data["Age"].mean()))
    wealth_features = np.array(data['Pclass'])
    gender_features = np.array(data["Sex"].replace('female',0).replace('male',1))
    name_length_features = np.array(data["Name"].apply(len))
    wealthy_men_features = gender_features * wealth_features
    corruption_features = (np.array(data["Fare"]))# * wealth_features
    col_mean = np.nanmean(corruption_features)
    indexes_of_nan = np.where(np.isnan(corruption_features))
    corruption_features[indexes_of_nan] = [col_mean for _ in range(len(indexes_of_nan))]

    features = np.concatenate((
        #name_length_features[np.newaxis],
        wealth_features[np.newaxis],
        age_features[np.newaxis],
        gender_features[np.newaxis],
        wealthy_men_features[np.newaxis],
        corruption_features[np.newaxis],
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
    scaler = StandardScaler()
    classifier = RandomForestClassifier(random_state=0)
    path = (getcwd() + r'\foursquare-location-matching')
    train_data, train_labels, test_data = get_data(path)
    train_features = get_features(train_data)
    z_scored_train_features = scaler.fit_transform(train_features)
    pred_rate = np.mean(cross_validate(z_scored_train_features, train_labels, 8, classifier))
    print(f"Prediciton rate is: {pred_rate*100}")
    trained_model = classifier.fit(z_scored_train_features, train_labels)
    test_features = get_features(test_data)
    z_scored_test_features = scaler.transform(test_features)
    prediciton = get_pred(trained_model, z_scored_test_features)
    save_model_for_evaluation(test_data,prediciton, path)




