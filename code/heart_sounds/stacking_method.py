from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold
from utils import caculate_MAcc
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, StratifiedKFold, GridSearchCV
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from mlxtend.classifier import StackingClassifier
import os
import pandas as pd
from utils import undersampling
from sklearn import svm
from random_forest import parameter_tuning_rf,model_testing
from SVM import parameter_tuning_svm
from Adaboost import parameter_tuning_ada
from GDBT import parameter_tuning_gbdt
import joblib

import warnings
warnings.filterwarnings("ignore")
args_out = '/home/deep/heart_science/result'


epochs = 1

def pro_to(pro_arr):
    not_sure = 0
    pre_list = []
    for i in range(len(pro_arr)):
        if np.abs(pro_arr[i][0]-pro_arr[i][1]) < 0.9:
            not_sure += 1
            pre_list.append(0)
        else:
            if pro_arr[i][0] > pro_arr[i][1]:
                pre_list.append(-1)
            else:
                pre_list.append(1)
    return pre_list,not_sure


def model_training_stack(x_train, y_train, cross_val, y_name, n_maj=None, n_min=None):
    sensitivity_list = []
    recall_list = []
    MAcc_list = []
    for epoch in range(epochs):
    # cross_val flag is used to specify if the model is used to test on a
    # cross validation set or the blind test set
        if cross_val:
            # splits the training data to perform 5-fold cross validation
            ss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=18*epoch)
            precision_list = [0]


            for train_index, test_index in ss.split(x_train, y_train):
                index = 0
                X_train = x_train[train_index]
                Y_train = y_train[train_index]
                X_test = x_train[test_index]
                Y_test = y_train[test_index]


                #invoke the parameter tuning functiom
                rf_params = parameter_tuning_rf(X_train, Y_train)
                svm_params = parameter_tuning_svm(X_train, Y_train)
                ada_params = parameter_tuning_ada(X_train, Y_train)

                #gdbt_params = parameter_tuning_gbdt(X,Y)

                rf = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'],
                                            n_jobs=-1,
                                            random_state=0)
                svr = svm.SVC(C=svm_params['C'], kernel='rbf', gamma=svm_params['gamma'], random_state=4, probability=True)

                ada = AdaBoostClassifier(n_estimators=ada_params['n_estimators'], learning_rate=ada_params['learning_rate'],
                                         algorithm='SAMME.R')

                #gdbt = GradientBoostingClassifier(learning_rate=gdbt_params['learning_rate'],n_estimators=gdbt_params['n_estimators'],
                                                     #max_depth=gdbt_params['max_depth'],subsample=gdbt_params['subsample'],random_state = 0)

                lr = LogisticRegression(C=1,max_iter=500)

                clfs = [ada,rf,svr]

                sclf = StackingClassifier(classifiers=clfs,use_probas=True,average_probas=False,
                                          meta_classifier=lr)
                sclf.fit(X_train,Y_train)

                # intialize the random forest classifier
                y_predict = sclf.predict(X_test)
                #pro = sclf.predict_proba(X_test)
                #y_predict,not_sure = pro_to(pro)
                #precision, recall, f_score, _ = precision_recall_fscore_support(Y_test, y_predict, pos_label=1,average='binary')
                c_mat = confusion_matrix(Y_test, y_predict)
                se,sp,MAcc = caculate_MAcc(c_mat)
                print(c_mat)
                precision_list.append(se)
                index += 1
                sensitivity_list.append(se)
                recall_list.append(sp)
                MAcc_list.append(MAcc)
                #uncertain_list.append(uncertain)

    print('------------------------------------------------------------------')
    print("best sensitivity is :",np.mean(sensitivity_list))
    print("best specifity is :",np.mean(recall_list))
    #print("uncertainty is :",np.mean(uncertain_list))
    print("best MACC is :",np.mean(MAcc_list))
    print('-------------------------------------------------------------------')
    #return sclf
    return np.mean(sensitivity_list),np.mean(recall_list),np.mean(MAcc_list)



if __name__ == "__main__":
    #feature_select_path = '/home/deep/heart_science/teacher.txt'
    model_apr = 'train-cross-val'
    train_file = '/Users/mac/Desktop/heart_science/data_feature1.csv'

    try:
        train_feature = pd.read_csv(train_file)
    except FileNotFoundError:
        print("无法找到此csv文件")
    train_feature.dropna(inplace=True)
    feature_list = train_feature.columns.values.tolist()
    feature_label = feature_list[-1]
    feature_list.remove(feature_label)

    '''
    if os.path.exists(feature_select_path):
        feature_list = []
        with open(feature_select_path, 'r+') as f:
            read_list = f.readlines()
            for item in read_list:
                feature_list.append(item.split('\n')[0])
    else:
        print('无法找到feature list，请重新运行特征选择')
    '''

    X_ = train_feature[feature_list]
    Y_ = train_feature[feature_label]
    n_maj = 0.47
    n_min = 1.0

    for i in range(8):

        X, Y = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                       maj_proportion=n_maj, min_proportion=n_min)
        num_positive = 0
        num_negitive = 0
        for i in range(len(Y)):
            if Y[i] == -1:
                num_negitive += 1
            else:
                num_positive += 1
        print("the number of negitive smaple is {0},and the number of positive sample is{1}".format(num_negitive,num_positive))



        sclf = model_training_stack(X,Y,True,'feature_label', n_maj, n_min)
        n_maj+=0.05


