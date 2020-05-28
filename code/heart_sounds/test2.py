from stacking_method import model_training_stack
from utils import undersampling
import os
import pandas as pd



if __name__ == '__main__':
    #path = '/home/deep/heart_science/feature_enhance'
    #data_subset = os.listdir(path)
    n_maj = 0.25
    n_min = 1.0
    epochs = 1

    for i in range(12):
        data_path = '/Users/mac/Desktop/heart_science/data_feature1.csv'
        train_data = pd.read_csv(data_path)
        train_data.dropna(inplace=True)
        feature_list = train_data.columns.values.tolist()
        feature_label = feature_list[-1]
        #feature_list0.remove(feature_label)
        #feature_list = feature_list0[:11] + feature_list0[12:13+i] + feature_list0[23:35] + feature_list0[36:37+i]
        #feature_list = feature_list0
        #print(feature_list)


        X_ = train_data[feature_list]
        Y_ = train_data[feature_label]
        X, Y = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                             maj_proportion=n_maj, min_proportion=n_min)
        sensitivity, recall, MACC = model_training_stack(X, Y, True, 'feature_label', n_maj, n_min)

        num_positive = 0
        num_negitive = 0
        for i in range(len(Y)):
            if Y[i] == -1:
                num_negitive += 1
            else:
                num_positive += 1
        print("the number of negitive smaple is {0},and the number of positive sample is{1}".format(num_negitive,
                                                                                                    num_positive))
        #n_maj += 0.05
        save_path = '/home/deep/heart_science/dicuss0.txt'
        with open(save_path, 'a+') as f:
            f.write('----------------------------------------------------------------')
            f.write('num mfcc is : ')
            f.write(str(i))
            f.write('\n')
            f.write('sensitivity is ')
            f.write(str(sensitivity))
            f.write('\n')
            f.write('reacall is ')
            f.write(str(recall))
            f.write('\n')
            f.write('MACC is ')
            f.write(str(MACC))

