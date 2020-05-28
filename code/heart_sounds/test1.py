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

n_maj = 0.25
n_min = 1.0
epochs = 1

def model_training_stack(x_train, y_train, cross_val, y_name, n_maj=None, n_min=None):
    last_precision = 0
    recall_list = []
    fscore_list = []
    MAcc_list = []
    precision_list = []
    for epoch in range(epochs):
    # cross_val flag is used to specify if the model is used to test on a
    # cross validation set or the blind test set
        if cross_val:
            # splits the training data to perform 5-fold cross validation
            ss = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=epoch*11)


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
                svr = svm.SVC(C=svm_params['C'], kernel='rbf', gamma=svm_params['gamma'], random_state=0, probability=True)

                ada = AdaBoostClassifier(n_estimators=ada_params['n_estimators'], learning_rate=ada_params['learning_rate'],
                                         algorithm='SAMME.R')

                #gdbt = GradientBoostingClassifier(learning_rate=gdbt_params['learning_rate'],n_estimators=gdbt_params['n_estimators'],
                                                     #max_depth=gdbt_params['max_depth'],subsample=gdbt_params['subsample'],random_state = 0)

                lr = LogisticRegression(C=1,max_iter=500)

                clfs = [svr]

                sclf = StackingClassifier(classifiers=clfs,use_probas=True,average_probas=False,
                                          meta_classifier=lr)
                sclf.fit(X_train,Y_train)
                # intialize the random forest classifier
                y_predict = sclf.predict(X_test)
                precision, recall, f_score, _ = precision_recall_fscore_support(Y_test, y_predict, pos_label=1,average='binary')
                c_mat = confusion_matrix(Y_test, y_predict)
                MAcc = caculate_MAcc(c_mat)
                #if precision > precision_list[index]:
                    #joblib.dump(sclf,'/home/deep/heart_science/model/sclf.model')
                precision_list.append(precision)
                recall_list.append(recall)
                fscore_list.append(f_score)
                MAcc_list.append(MAcc)
                index += 1


        '''
        if np.mean(precision_list) > last_precision:
            print(precision_list)
            last_precision = np.mean(precision_list)
            print('best precision is:',np.mean(precision_list))
            print('best recall is',np.mean(recall_list))
            print('best f-score is',np.mean(fscore_list))
            print('best MAcc is',np.mean(MAcc_list))
        '''


    #return sclf
    return np.mean(precision_list),np.mean(recall_list),np.mean(fscore_list),np.mean(MAcc_list)



if __name__ == "__main__":
    path = '/home/deep/heart_science/expriment3'
    feature_subset = os.listdir(path)
    print(feature_subset)
    for i in range(len(feature_subset)):
        feature_select_path = '/home/deep/heart_science/expriment3/' + str(feature_subset[i])
        model_apr = 'train-cross-val'
        train_file = '/Users/mac/Desktop/heart_science/data_feature1.csv'
        test_file = '/home/deep/heart_science/test_data.csv'

        try:
            train_feature = pd.read_csv(train_file)
        except FileNotFoundError:
            print("无法找到此csv文件")
        train_feature.dropna(inplace=True)
        # train_feature = train_feature.sample(frac=1).reset_index(drop=True)
        # test_feature = test_feature.sample(frac=1).reset_index(drop=True)
        print(feature_select_path)
        if os.path.exists(feature_select_path):
            feature_list = []
            with open(feature_select_path, 'r+') as f:
                read_list = f.readlines()
                for item in read_list:
                    feature_list.append(item.split('\n')[0])
        else:
            print('无法找到feature list，请重新运行特征选择')

        X_ = train_feature[feature_list]
        Y_ = train_feature['feature_label']
        print(len(Y_))
        X, Y = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                       maj_proportion=n_maj, min_proportion=n_min)
        num_positive = 0
        num_negitive = 0
        for j in range(len(Y)):
            if Y[j] == -1:
                num_negitive += 1
            else:
                num_positive += 1
        print("the number of negitive smaple is {0},and the number of positive sample is{1}".format(num_negitive,num_positive))



        sensitivity,recall,fscore,MACC = model_training_stack(X,Y,True,'feature_label', n_maj, n_min)
        save_path = '/home/deep/heart_science/result3/' + str(feature_subset[i])
        with open(save_path,'w+') as f:
            f.write('sensitivity is ')
            f.write(str(sensitivity))
            f.write('\n')
            f.write('reacall is ')
            f.write(str(recall))
            f.write('\n')
            f.write('fscore is ')
            f.write(str(fscore))
            f.write('\n')
            f.write('MACC is ')
            f.write(str(MACC))


'''
['mean_s1_time','std_s1_time','mean_systole_time','std_systole_time',\
                        'mean_s2_time','std_s2_time','mean_diastole_time','std_diastole_time',\
                        'mean_ratio_s1_systole','std_ratio_s1_systole','mean_ratio_s2_diastole','std_ratio_s2_diastole',\
                        'mean_s1_skew','std_s1_skew','mean_systole_skew','std_systole_skew','mean_s2_skew','std_s2_skew',\
                        'mean_diastole_skew','std_diastole_skew','mean_s1_kurtosis','std_s1_kurtosis','mean_systole_kurtosis',\
                        'std_systole_kurtosis','mean_s2_kurtosis','std_s2_kurtosis','mean_diastole_kurtosis','std_diastole_kurtosis',\
                        'mean_s1_power','std_s1_power','mean_systole_power','std_systole_power','mean_s2_power','std_s2_power', \
                        'mean_diastole_power','std_diastole_power','mfcc_s1_1','mfcc_s1_2','mfcc_s1_3','mfcc_s1_4','mfcc_s1_5',\
                        'mfcc_s1_6','mfcc_s1_7','mfcc_s1_8','mfcc_s1_9','mfcc_s1_10','mfcc_s1_11','mfcc_s1_12', \
                        'mfcc_systole_1','mfcc_systole_2','mfcc_systole_3','mfcc_systole_4','mfcc_systole_5','mfcc_systole_6',\
                        'mfcc_systole_7','mfcc_systole_8','mfcc_systole_9','mfcc_systole_10','mfcc_systole_11','mfcc_systole_12',\
                        'mfcc_s2_1','mfcc_s2_2','mfcc_s2_3','mfcc_s2_4','mfcc_s2_5','mfcc_s2_6','mfcc_s2_7','mfcc_s2_8','mfcc_s2_9', \
                        'mfcc_s2_10','mfcc_s2_11','mfcc_s2_12', \
                        'mfcc_diastole_1','mfcc_diastole_2','mfcc_diastole_3','mfcc_diastole_4','mfcc_diastole_5','mfcc_diastole_6', \
                        'mfcc_diastole_7','mfcc_diastole_8','mfcc_diastole_9','mfcc_diastole_10','mfcc_diastole_11','mfcc_diastole_12']
'''


