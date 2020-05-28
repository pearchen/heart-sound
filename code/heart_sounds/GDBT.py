from sklearn.ensemble import GradientBoostingClassifier
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
import joblib
import pandas as pd
from utils import *
from sklearn.metrics import confusion_matrix

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


def parameter_tuning_gbdt(x,y):
    parameters = [{'learning_rate':[0.1,0.15,0.2,0.25],'n_estimators':[100,300,500],'max_depth':[10,30,50,70],
                   'subsample':[0.8,0.9,1.0]}]

    cvs = StratifiedKFold(5)
    gbdt = GradientBoostingClassifier(random_state = 10)
    gsvm = GridSearchCV(estimator=gbdt,param_grid=parameters,n_jobs=8,cv = list(cvs.split(x,y)))

    gsvm.fit(x,y)
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
        gbdt_params = parameter_tuning_gbdt(training_data, training_label)##################

        # intialize the random forest classifier
        feat_select = GradientBoostingClassifier(learning_rate=gbdt_params['learning_rate'],n_estimators=gbdt_params['n_estimators'],
                                                 max_depth=gbdt_params['max_depth'],random_state = 0)
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

def model_training_gbdt(x_train,y_train,cross_val,y_name, n_maj=None, n_min=None):
    if cross_val:
        ss = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=0)

        pr_list = []
        re_list = []
        fs_list = []

        for train_index,test_index in ss.split(x_train,y_train):
            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_test = x_train[test_index]
            Y_test = y_train[test_index]

            gbdt_params = parameter_tuning_gbdt(X_train,Y_train)
            print(Y_train)
            gbdt = GradientBoostingClassifier(learning_rate=gbdt_params['learning_rate'],n_estimators=gbdt_params['n_estimators'],
                                                 max_depth=gbdt_params['max_depth'],subsample=gbdt_params['subsample'],random_state = 0)
            gbdt.fit(X_train,Y_train)
            joblib.dump(gbdt,'/Users/mac/Desktop/heart_science/model/gbdt.model')

            y_predicted = gbdt.predict(X_test)
            print(y_predicted)

            pr, re, fs, _ = precision_recall_fscore_support(Y_test, y_predicted, pos_label=1, average='binary')
            pr_list.append(pr)
            re_list.append(re)
            fs_list.append(fs)

        return gbdt

    else:
        svm_params = parameter_tuning_gbdt(x_train,y_train)

        svr = svm.SVC(C = svm_params['C'],kernel = 'rbf',gamma = svm_params['gamma'],class_weight = 'balanced',random_state = 0,probability=True)
        svr.fit(x_train,y_train)

        return svr

def model_testing(rf_model, x_test, y_test, output_folder):

    # test the random forest classifier
    predicted_labels = rf_model.predict(x_test)

    # gather the prediction probabilities used to plot precision recall curves
    pred_probab = rf_model.predict_proba(x_test)

    # compute precision, recall and fscore
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, predicted_labels, pos_label=1, average='binary')

    # compute the confusion matrix
    c_mat = confusion_matrix(y_test, predicted_labels)
    print(c_mat)

    MAcc = caculate_MAcc(c_mat)

    # plot the confusion matrix and precision recall curves
    plot_confusion_matrix(c_mat, ['Normal', 'Abnormal'], output_folder)
    plot_precision_recall_curve(y_test, pred_probab, output_folder)

    return precision, recall, fscore ,MAcc

if __name__ == "__main__":
    feature_select_path = '/Users/mac/Desktop/heart_science/feature_select.txt'
    model_apr = 'train-cross-val'
    train_file = '/Users/mac/Desktop/heart_science/train_data.csv'
    test_file = '/Users/mac/Desktop/heart_science/test_data.csv'

    try:
        train_feature = pd.read_csv(train_file)
        test_feature = pd.read_csv(test_file)
    except FileNotFoundError:
        print("无法找到此csv文件")
    train_feature.dropna(inplace=True)
    # train_feature = train_feature.sample(frac=1).reset_index(drop=True)
    test_feature.dropna(inplace=True)
    # test_feature = test_feature.sample(frac=1).reset_index(drop=True)

    if os.path.exists(feature_select_path):
        feature_list = []
        with open(feature_select_path, 'r+') as f:
            read_list = f.readlines()
            feature_list = read_list[0].split(',')
    else:
        print('无法找到feature list，请重新运行特征选择')

    X_ = train_feature[feature_list]
    Y_ = train_feature['feature_label']
    X_test = test_feature[feature_list]
    Y_test = test_feature['feature_label']
    if model_apr == 'train-cross-val':

        # Undersampling is used in situations where one of the data among the different classes is highly imbalanced.
        # invoke the undersampling module that sub-samples the majority class and returns nearly balanced data.
        X, Y = undersampling(X_.values, Y_.values, majority_class=0, minority_class=1, maj_proportion=n_maj,
                             min_proportion=n_min)
        x_test, y_test = undersampling(X_test.values, Y_test.values, majority_class=0, minority_class=1,
                                       maj_proportion=n_maj, min_proportion=n_min)

        # invoke the training module
        svr = model_training_gbdt(X, Y, True, 'feature_label', n_maj, n_min)
        precision, recall, fscore, MAcc_score = model_testing(svr, x_test, y_test, args_out)
        print("Results on 5-fold Cross Validation Set")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", fscore)
        print("MAcc", MAcc_score)

    elif model_apr == 'train-test':
        # test_size = 0.2

        # split the data into non-overlapping train and test instances
        # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

        # invoke the undersampling module
        x_test, y_test = undersampling(X_test.values, Y_test.values, majority_class=-1, minority_class=1,
                                       maj_proportion=n_maj, min_proportion=n_min)

        # invoke the training module
        svm = joblib.load('/Users/mac/Desktop/heart_science/model/gbdt.model')

        # invoke the test module
        precision, recall, fscore, MAcc = model_testing(svm, x_test, y_test,args_out)
        print("Results on the Test Set")
        print("Precison: ", precision)
        print("Recall: ", recall)
        print("F-score: ", fscore)
        print("MAcc",MAcc)


