import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
import csv
import copy

import numpy as np
from random import randrange
from utils import undersampling

from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize
from stacking_method import model_training_stack

import warnings
warnings.filterwarnings("ignore")

n_maj = 0.35
n_min = 1.0
epochs = 1
periods = 3


#计算特征和类的平均值
def calcMean(x,y):

    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

#计算Pearson系数
def calcPearson(x_,y_):
    x = np.array(list(i for i in x_)).tolist()[0]  # 这两行都是转换格式
    y = np.array(list(i for i in y_)).tolist()[0]
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p
#冗余性计算  feature_a表示待检测特征  feature_sel代表已选择的特征集合
def cal_rongyu(feature_a,feature_sel,feature_num):
    pearson_sum = 0
    feature_a = feature_a.values.tolist()
    feature_a = list(np.reshape(feature_a,(1,len(feature_a))))
    feature_sel = feature_sel.values.tolist()
    feature_sel = list(np.reshape(feature_sel,(feature_num,len(feature_sel))))
    for i in range(len(feature_sel)):
        pearson = calcPearson(feature_a,feature_sel)
        pearson_sum = pearson_sum + pearson
    rf = pearson_sum / len(feature_sel)
    return rf

def distanceNorm(Norm, D_value):
    # initialization


    # Norm for distance
    if Norm == '1':
        counter = np.absolute(D_value)
        counter = np.sum(counter)
    elif Norm == '2':
        counter = np.power(D_value, 2)
        counter = np.sum(counter)
        counter = np.sqrt(counter)
    elif Norm == 'Infinity':
        counter = np.absolute(D_value)
        counter = np.max(counter)
    else:
        raise Exception('We will program this later......')

    return counter


def fit(features, labels, iter_ratio):
    # initialization
    features = features.values
    labels = list(labels)
    (n_samples, n_features) = np.shape(features)
    print(n_features)
    distance = np.zeros((n_samples, n_samples))
    weight = np.zeros(n_features)

    if iter_ratio >= 0.5:
        # compute distance
        for index_i in range(n_samples):
            for index_j in range(index_i + 1, n_samples):
                D_value = features[index_i] - features[index_j]
                distance[index_i, index_j] = distanceNorm('2', D_value)
        distance += distance.T
    else:
        pass;

    # start iteration
    for iter_num in range(int(iter_ratio * n_samples)):
        # print iter_num;
        # initialization
        nearHit = list()
        nearMiss = list()
        distance_sort = list()

        # random extract a sample
        index_i = randrange(0, n_samples, 1)
        self_features = features[index_i]

        # search for nearHit and nearMiss
        if iter_ratio >= 0.5:
            distance[index_i, index_i] = np.max(distance[index_i])  # filter self-distance
            for index in range(n_samples):
                distance_sort.append([distance[index_i, index], index, labels[index]])
        else:
            # compute distance respectively
            distance = np.zeros(n_samples)
            for index_j in range(n_samples):
                D_value = features[index_i] - features[index_j]
                distance[index_j] = distanceNorm('2', D_value)
            distance[index_i] = np.max(distance) # filter self-distance
            for index in range(n_samples):
                distance_sort.append([distance[index], index, labels[index]])
        distance_sort.sort(key=lambda x: x[0])
        for index in range(n_samples):
            if nearHit == [] and distance_sort[index][2] == labels[index_i]:
                # nearHit = distance_sort[index][1];
                nearHit = features[distance_sort[index][1]]
            elif nearMiss == [] and distance_sort[index][2] != labels[index_i]:
                # nearMiss = distance_sort[index][1]
                nearMiss = features[distance_sort[index][1]]
            elif nearHit != [] and nearMiss != []:
                break
            else:
                continue

        # update weight
        weight = weight - np.power(self_features - nearHit, 2) + np.power(self_features - nearMiss, 2)
    return weight / (iter_ratio * n_samples)

'''
def test():
    (features, labels) = make_classification(n_samples=500)
    features = normalize(X=features, norm='l2', axis=0)
    for x in range(1, 10):
        weight = fit(features, labels, 1)
    print(weight)
'''

def cal_relief(features,labels):
    weight = fit(features,labels,1)
    return weight



def data_straggling(data,k,row_name):
    data = np.array(data)
    kmodel = KMeans(n_clusters=k,n_jobs=-1)
    kmodel.fit(data.reshape((len(data),1)))
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean().iloc[1:]
    w = [0] + list(w[0]) + [data.max()]
    d3 = pd.cut(data,w,labels=range(k))
    return d3

def normalized_option(row_data):
    data_max = np.max(row_data)
    data_min = np.min(row_data)
    op_data = (row_data - data_min)/(data_max - data_min)
    return op_data

def data_normalized(frame_data,feature_name):
    for item in feature_name:
        frame_data[item] = normalized_option(frame_data[item])
    return frame_data

# 互信息
def cal_nmi(A,B):
    # len(A) should be equal to len(B)
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

    MIhat = 2.0*MI/(HX+HY)
    return MIhat

# 信息熵
def ent(data):
    prob1 = pd.value_counts(data) / len(data)
    return sum(np.log2(prob1) * prob1 * (-1))


# 信息增益
def cal_gain(data, str1, str2):
    e1 = data.groupby(str1).apply(lambda x: ent(x[str2]))
    p1 = pd.value_counts(data[str1]) / len(data[str1])
    e2 = sum(e1 * p1)
    return ent(data[str2]) - e2

def BDS(feature_data,good_feature,feature_score):
    last_precisionf = 0
    last_precisionb = 0
    continue_flag = True
    feature_A = copy.copy(good_feature)
    feature_A.remove(good_feature[0])
    forward_sub = [good_feature[0]]                   #前向已选择特征集合
    backward_sub = copy.copy(good_feature)            #后向已选择特征集合
    backward_sub.remove(good_feature[-1])
    moved_list = []
    moved_list.append(good_feature[-1])               #后向已去除特征集合
    score_list1 = copy.copy(feature_score)
    score_list1.remove(score_list1[0])


    while continue_flag:

        ipt_degree1 = {}
        ipt_degree2 = {}
        for i in range(feature_A):
            str_feature = []
            str_feature.append(str(feature_A[i]))
            Df = score_list1[i] / cal_rongyu(feature_data[str_feature], feature_data[forward_sub], len(forward_sub))
            ipt_degree1[feature_A[i]] = Df
        sort1 = sorted(zip(ipt_degree1.values(), ipt_degree1.keys()), reverse=True)
        print(sort1)
        for i in range(backward_sub):
            str_feature = []
            str_feature.append(str(feature_A[i]))
            Df = score_list1[i] / cal_rongyu(feature_data[str_feature], feature_data[moved_list], len(moved_list))
            ipt_degree2[backward_sub] = Df
        sort2 = sorted(zip(ipt_degree2.values(), ipt_degree2.keys()), reverse=True)
        print(sort2)

        if sort1[0][1] not in moved_list:
            try_feature = []
            try_feature.append(sort1[0][1])
            test_feature1 = try_feature + forward_sub
            feature_A.remove(try_feature)

            X_ = feature_data[test_feature1]
            Y_ = feature_data['feature_label']
            X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                             maj_proportion=n_maj, min_proportion=n_min)

            now_precision, _, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)

            if now_precision > last_precisionf:
                last_precisionf = now_precision
                forward_sub.append(try_feature)
            else:
                pass
        else:
            feature_A.remove(sort1[0][1])


        if sort2[0][1] not in forward_sub:
            try_feature = sort2[0][1]
            test_feature2 = backward_sub.remove(try_feature)
            moved_list.append(try_feature)

            X_ = feature_data[test_feature2]
            Y_ = feature_data['feature_label']
            X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                             maj_proportion=n_maj, min_proportion=n_min)

            now_precision, _, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)

            if now_precision > last_precisionb:
                last_precisionb = now_precision
                backward_sub.remove(try_feature)
            else:
                pass

            if backward_sub == forward_sub:
                continue_flag = False
        else:
            pass

    return backward_sub





#动态搜索增GSFS算法
def GSFS(feature_data,good_feature,feature_score):
    continue_flag = True
    last_precision = 0
    count_miss = 0
    L = 2
    score_list = feature_score
    feature_A = good_feature            #待选特征
    feature_B = []                      #已选特征
    best_feature = feature_A[score_list.index(np.max(score_list))]
    feature_B.append(best_feature)
    feature_A.remove(best_feature)
    score_list.remove(np.max(score_list))
    while continue_flag:
        ipt_degree = {}  # 从相关性和冗余度两个方面来衡量候选特征的重要程度,score有正负 是否要绝对值？
        for i in range(len(feature_A)):
            str_feature = []
            str_feature.append(str(feature_A[i]))
            Df = score_list[i] / cal_rongyu(feature_data[str_feature],feature_data[feature_B],len(feature_B))
            ipt_degree[feature_A[i]] = Df
        sort = sorted(zip(ipt_degree.values(),ipt_degree.keys()),reverse = True)
        print(sort)
        if len(sort) > 1:
            buffer = list(sort[i][1] for i in range(L))
            for i in range(len(buffer)):
                for j in range(i + 1, len(buffer)):
                    try_feature = []
                    if i == j:
                        continue
                    try_feature.append(buffer[j])
                    try_feature.append(buffer[i])  # 将try_feature送入模型训练，保留效果最好的那一组try_feature
                    test_feature = feature_B + try_feature

                    print(test_feature)
                    X_ = feature_data[test_feature]
                    Y_ = feature_data['feature_label']
                    X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                                     maj_proportion=n_maj, min_proportion=n_min)

                    now_precision, _, _, = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)
                    if now_precision > last_precision:
                        saved_feature = try_feature
                        last_precision = now_precision
                    else:
                        pass
                        #buffer里现在有L个特征了，需要根据分类模型的效果去除R个
        else:
            buffer = list(sort[i][1] for i in range(1))
            for i in range(len(buffer)):
                    try_feature = []
                    try_feature.append(buffer[i])  # 将try_feature送入模型训练，保留效果最好的那一组try_feature
                    test_feature = feature_B + try_feature

                    print(test_feature)
                    X_ = feature_data[test_feature]
                    Y_ = feature_data['feature_label']
                    X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                                     maj_proportion=n_maj, min_proportion=n_min)

                    now_precision, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)
                    if now_precision > last_precision:
                        saved_feature = try_feature
                        last_precision = now_precision
                    else:
                        pass



        print("此轮迭代完毕！最好的precision是",last_precision)
        print("添加了属性",saved_feature)

        feature_B += saved_feature
        if len(buffer) == 2:
            feature_A.remove(buffer[0])
            feature_A.remove(buffer[1])
        else:
            feature_A.remove(buffer[0])
        saved_feature = []

        if len(buffer) < 2:
            continue_flag = False

    return feature_B





#动态搜索增L去R算法
def LRS(feature_data,good_feature,feature_score):
    for period in range(periods):
        print('it is number ',period)
        if period == 0:
            continue_flag = True
            best_precision = 0
            L = 3
            score_list = copy.copy(feature_score)
            feature_A = copy.copy(good_feature)            #待选特征
            feature_B = []                      #已选特征
            best_feature = feature_A[score_list.index(np.max(score_list))]
            feature_B.append(best_feature)
            feature_A.remove(best_feature)
            score_list.remove(np.max(score_list))
        else:
            continue_flag = True
            score_list = copy.copy(feature_score)
            feature_A = copy.copy(good_feature)
            for x in range(len(feature_B)):
                print(feature_B[x])
                position = feature_A.index(feature_B[x])
                print(position)
                del score_list[position]
                del feature_A[position]


            #score_list.remove(score_list[x] for x in feature_A.index(feature_B))
            #feature_A.remove(feature_B)
        print(len(score_list))
        print(len(feature_A))
        while continue_flag:
            ipt_degree = {}  # 从相关性和冗余度两个方面来衡量候选特征的重要程度,score有正负 是否要绝对值？
            for i in range(len(feature_A)):
                str_feature = []
                str_feature.append(str(feature_A[i]))
                Df = score_list[i] / cal_rongyu(feature_data[str_feature],feature_data[feature_B],len(feature_B))
                ipt_degree[feature_A[i]] = Df
            sort = sorted(zip(ipt_degree.values(),ipt_degree.keys()),reverse = True)
            print(sort)
            if len(sort) >= 3:
                buffer = list(sort[i][1] for i in range(L))                #buffer里现在有L个特征了，需要根据分类模型的效果去除R个
            else:
                buffer = list(sort[i][1] for i in range(len(sort)))
            #########################################################################################
            for i in range(len(buffer)):
                try_feature_1 = list(idx for idx in buffer)
                test_feature_1 = list(feature_B[ii] for ii in range(len(feature_B)))
                test_feature_1 = test_feature_1 + try_feature_1
                index1 = int(i + len(feature_B))
                del test_feature_1[index1]

                X_ = feature_data[test_feature_1]
                Y_ = feature_data['feature_label']
                X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                                 maj_proportion=n_maj, min_proportion=n_min)
                now_precision, _, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)

                if now_precision > best_precision:
                    option = try_feature_1
                    del option[i]
                    saved_feature = option
                    best_precision = now_precision

                #########加入减去上一轮减去的特征的基础上再减去一个
                for j in range(len(try_feature_1) - i):
                    if i == j:
                        continue
                    else:
                        try_feature_2 = list(idx for idx in try_feature_1)
                        test_feature_2 = list(feature_B[ii] for ii in range(len(feature_B)))
                        test_feature_2 = test_feature_2 + try_feature_2
                        index2 = int(j + index1)
                        del test_feature_2[index2 - 1]


                        X_ = feature_data[test_feature_2]
                        Y_ = feature_data['feature_label']
                        X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                                         maj_proportion=n_maj, min_proportion=n_min)

                        now_precision, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)

                        if now_precision > best_precision:
                            option = try_feature_2
                            del option[j]
                            saved_feature = option
                            best_precision = now_precision



                    for k in range(len(try_feature_2) - j):
                        if i == j or i == k or j == k:
                            continue
                        else:
                            try_feature_3 = list(idx for idx in try_feature_2)
                            test_feature_3 = list(feature_B[ii] for ii in range(len(feature_B)))
                            test_feature_3 = test_feature_3 + try_feature_3
                            del test_feature_3[int(k + index2) - 1]

                            X_ = feature_data[test_feature_3]
                            Y_ = feature_data['feature_label']
                            X_train, Y_train = undersampling(X_.values, Y_.values, majority_class=-1, minority_class=1,
                                                             maj_proportion=n_maj, min_proportion=n_min)

                            now_precision, _, _, _ = model_training_stack(X_train, Y_train, True, 'feature_label', n_maj, n_min)

                            if now_precision > best_precision:
                                option = try_feature_3
                                del option[k]
                                saved_feature = option
                                best_precision = now_precision

            print("此轮迭代完毕！最好的precision是",best_precision)
            print("添加了属性",saved_feature)

            feature_B += saved_feature
            for item in buffer:
                feature_A.remove(item)
            print(feature_B)
            saved_feature = []

            if len(buffer) < 3:
                continue_flag = False

    return feature_B


if __name__ == "__main__":


    feature_select_path = '/home/deep/heart_science/data_feature0.csv'
    iter_numbers = 10
    best_feature_list = []
    try:
        data_feature = pd.read_csv(feature_select_path)
    except FileNotFoundError:
        print("无法找到此csv文件")
    data_feature.dropna(inplace=True)
    columns = data_feature.columns.values.tolist()
    columns.remove('feature_label')
    start_feature = data_normalized(data_feature[columns],columns)
    frame = []
    for i in columns:
        frame.append(data_straggling(start_feature[i],5,i))
    frame_feature = pd.DataFrame(frame,columns = columns)
    frame_label = data_feature['feature_label']
    start_feature = pd.concat((frame_feature,frame_label),axis=1)
    start_feature = pd.DataFrame(start_feature)
    print(start_feature)
    start_feature.to_csv('/home/deep/heart_science/data_feature1.csv',index= False)


    mni_list = []
    gain_list = []
    relief_list = []
    select_feature = []
    select_score = []
    feature_select_path0 = '/home/deep/heart_science/data_feature2.csv'
    try:
        data_feature = pd.read_csv(feature_select_path0)
    except FileNotFoundError:
        print("无法找到此csv文件")
    data_feature.dropna(inplace=True)
    columns = data_feature.columns.values.tolist()
    columns.remove('feature_label')
    start_feature = data_feature[columns]

    for i in columns:
        mni = cal_nmi(start_feature[i],data_feature['feature_label'])
        gain = cal_gain(data_feature,i,'feature_label')
        mni_list.append(mni)
        gain_list.append(gain)
    relief_list = cal_relief(start_feature,data_feature['feature_label'])
    mni_list = normalized_option(mni_list)
    gain_list = normalized_option(gain_list)
    relief_list = normalized_option(relief_list)
    total_score = mni_list + gain_list + relief_list

    score_zip = zip(list(columns),total_score)

    for name, score in score_zip:
        if score >= 0.4:
            select_feature.append(name)
            select_score.append(score)



    final_feature = list(GSFS(data_feature,select_feature,select_score))
    with open('/home/deep/heart_science/feature_select0.txt','w+') as f:
        for i in range(len(final_feature)):
            f.write(final_feature[i])
            f.write('\n')






