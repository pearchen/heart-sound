import os
import numpy as np
import pandas as pd

def normalize_option(row_data,op):
    if op == 1:
        data_max = np.max(row_data)
        data_min = np.min(row_data)
        op_data = (row_data - data_min) / (data_max - data_min)
    elif op == 2:
        data_std = np.std(row_data)
        data_mean = np.mean(row_data)
        op_data = (row_data - data_mean) / data_std
    elif op == 3:
        op_data = []
        row_data = row_data.tolist()
        for i in range(len(row_data)):
            op_data.append(1.0/(1+np.exp(-float(row_data[i]))))
    return op_data

def data_normalized(data_feature,feature_name,op):
    for i in range(len(feature_name)):
        data_feature[feature_name[i]] = normalize_option(data_feature[feature_name[i]],op)
    return data_feature

if __name__ == '__main__':
    data_path = '/Users/mac/Desktop/heart_science/data_feature0.csv'
    train_data = pd.read_csv(data_path)
    train_data.dropna(inplace=True)
    feature_list0 = train_data.columns.values.tolist()
    feature_label = feature_list0[-1]
    feature_list0.remove(feature_label)

    nor_data = data_normalized(train_data,feature_list0,1)
    df_feat_ext = pd.DataFrame(nor_data)

    out_file = '/Users/mac/Desktop/heart_science/data_feature1.csv'
    try:
        df_feat_ext.to_csv(out_file, index=False)
    except Exception:
        print("Output path does not exist")