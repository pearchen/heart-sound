from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd
from utils import *
from sklearn.externals import joblib
import csv
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


n_maj = 0.25
n_min = 1.0
args_out = '/Users/mac/Desktop/heart_science/result'

def normalize_option(row_data):
    data_max = np.max(row_data)
    data_min = np.min(row_data)
    op_data = (row_data - data_min) / (data_max - data_min)
    return op_data


def data_normalized(data_feature,feature_name):
    for i in range(len(feature_name)):
        data_feature[feature_name[i]] = normalize_option(data_feature[feature_name[i]])
    return data_feature



def parameter_tuning(x, y):                 #随机森林调参
    # specify the search space for paramters you want to optimize
    estimators = np.arange(10, 60, 20)
    m_depth = np.arange(1, 10, 2)

    # use stratified K fold to test the optimization results
    cvs = StratifiedKFold(5)
    params = [{'n_estimators': estimators, 'max_depth': m_depth}]

    # intialize a random forest classifier
    rf = RandomForestClassifier(class_weight='balanced', random_state=0)


    # search for the optimal set of parameters for Random Forest Classifier
    gsvm = GridSearchCV(estimator=rf, param_grid=params, n_jobs=-1, cv=list(cvs.split(x, y)))
    gsvm.fit(x, y)
    return gsvm.best_params_

def feature_selection(data_frame, y_name):       # data_frame是特征  y_name是label

    feature_list = []
    columns = data_frame.columns.values.tolist()
    columns.remove(y_name)

    # run the feature selection for 50 times
    for i in range(50):
        print(i)
        # shuffle the data on each iteration
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)          #相当于给每一行一个index

        training_data = data_frame[columns]
        training_label = data_frame[y_name]

        # invoke the paramter tuning function
        rf_params = parameter_tuning(training_data, training_label)

        # intialize the random forest classifier
        feat_select = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], class_weight='balanced', random_state=0)
        feat_select.fit(training_data, training_label)
        #print feature_subset.feature_importances_

        # extract the feature importances and sort them in the descending order of their importance score
        features = sorted(zip(map(lambda x: round(x, 4), feat_select.feature_importances_), columns),
                     reverse=True)         #特征重要度由大到小排序

        sc, f_names = zip(*features)

        # extract top 30 features as informative ones
        feature_list.append(set(list(f_names[:25])))

    # take the union of the different feature selection runs
    feat_union = list(set.union(*feature_list))                #set.union相同的元素只会出现一次

    print("Number of features selected: ", len(feat_union))
    return feat_union



if __name__ == "__main__":
    feature_select_path = '/home/deep/heart_science/feature_select.txt'
    model_apr = 'train-cross-val'
    try:
        df_feature = pd.read_csv('/home/deep/heart_science/data_feature0.csv')
    except FileNotFoundError:
        print("无法找到此csv文件")
    df_feature.dropna(inplace=True)
    #df_feature = df_feature.sample(frac=1).reset_index(drop=True)
    columns = df_feature.columns.values.tolist()
    columns.remove('feature_label')

    df_feature[columns] = df_feature[columns]/10
    df_feature.to_csv('/home/deep/heart_science/data_feature0.csv',index= False)

