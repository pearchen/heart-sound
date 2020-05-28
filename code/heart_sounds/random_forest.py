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



def parameter_tuning_rf(x, y):                 #随机森林调参
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
        rf_params = parameter_tuning_rf(training_data, training_label)

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


def model_training_rf(x_train, y_train, cross_val, y_name, n_maj=None, n_min=None):

    # cross_val flag is used to specify if the model is used to test on a
    # cross validation set or the blind test set
    if cross_val:
        # splits the training data to perform 5-fold cross validation
        ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        pr_list = []
        re_list = []
        fs_list = []

        for train_index, test_index in ss.split(x_train, y_train):
            X_train = x_train[train_index]
            Y_train = y_train[train_index]
            X_test = x_train[test_index]
            Y_test = y_train[test_index]

            #invoke the parameter tuning functiom
            rf_params = parameter_tuning_rf(X_train, Y_train)

            # intialize the random forest classifier
            rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], n_jobs=-1, random_state=0)
            rfc.fit(X_train, Y_train)
            joblib.dump(rfc, '/home/deep/heart_science/model/rf.model')

            # use the classifier to predict on cross val set
            y_predicted = rfc.predict(X_test)

            # compute the precision, recall and fscore
            pr, re, fs, _ = precision_recall_fscore_support(Y_test, y_predicted, pos_label=1, average='binary')
            pr_list.append(pr)
            re_list.append(re)
            fs_list.append(fs)
            if pr >= 0.89:
                joblib.dump(rfc, '/home/deep/heart_science/model/sclf.model')
            print(pr)

        return rfc

    else:
        # This section only trains the model which is used to test on the blind test set
        rf_params = parameter_tuning_rf(x_train, y_train)
        rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], n_jobs=-1, random_state=0)
        rfc.fit(x_train, y_train)

        return rfc

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
    #plot_confusion_matrix(c_mat, ['Normal', 'Abnormal'], output_folder)
    #plot_precision_recall_curve(y_test, pred_probab, output_folder)

    return precision, recall, fscore ,MAcc

if __name__ == "__main__":
    feature_select_path = '/home/deep/heart_science/feature_select.txt'
    model_apr = 'train-cross-val'
    train_file = '/home/deep/heart_science/train_data.csv'
    test_file = '/home/deep/heart_science/test_data.csv'

    try:
        train_feature = pd.read_csv(train_file)
        test_feature = pd.read_csv(test_file)
    except FileNotFoundError:
        print("无法找到此csv文件")
    train_feature.dropna(inplace=True)
    #train_feature = train_feature.sample(frac=1).reset_index(drop=True)
    test_feature.dropna(inplace=True)
    #test_feature = test_feature.sample(frac=1).reset_index(drop=True)

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
    print(Y_)


    if model_apr == 'train-cross-val':

        # Undersampling is used in situations where one of the data among the different classes is highly imbalanced.
        # invoke the undersampling module that sub-samples the majority class and returns nearly balanced data.
        X, Y = undersampling(X_.values, Y_.values, majority_class=0, minority_class=1, maj_proportion=n_maj, min_proportion=n_min)
        x_test, y_test = undersampling(X_test.values, Y_test.values, majority_class=0, minority_class=1,
                                       maj_proportion=n_maj, min_proportion=n_min)
        # invoke the training module
        rf = model_training_rf(X, Y, True, 'feature_label', n_maj, n_min)
        #rf = joblib.load('/home/deep/heart_science/model/rf.model')
        precision, recall, fscore, MAcc_score = model_testing(rf, x_test, y_test, args_out)
        print("Results on 5-fold Cross Validation Set")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", fscore)
        print("MAcc", MAcc_score)

    elif model_apr == 'train-test':
        #test_size = 0.2

        # split the data into non-overlapping train and test instances
        #x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

        # invoke the undersampling module
        x_test, y_test = undersampling(X_test.values, Y_test.values, majority_class=-1, minority_class=1, maj_proportion=n_maj, min_proportion=n_min)
        # invoke the training module
        rf = joblib.load('/Users/mac/Desktop/heart_science/model/rf.model')

        # invoke the test module
        precision, recall, fscore, MAcc_score= model_testing(rf, x_test, y_test, args_out)
        print("Results on the Test Set")
        print("Precison: ", precision)
        print("Recall: ", recall)
        print("F-score: ", fscore)
        print("MAcc_score",MAcc_score)


