import numpy as np
import csv
import os
import wave
import librosa
import math
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fftpack import dct


NFFT = 256

def mel_coefficients(sample_rate, nfilt, pow_frames,num_cof):
    low_freq_mel = 0
    num_mel_coeff = num_cof
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2.0) / 700.0))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc = dct(filter_banks, type=2, axis=0, norm='ortho')[1:(num_mel_coeff+1)]
    (ncoeff,) = mfcc.shape
    cep_lifter = ncoeff
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    return mfcc

def frequency_features(data_sequence,num_cof):

    # computes the power spectrum of the signal
    hamming_distance = data_sequence*np.hamming(len(data_sequence))
    fft = np.absolute(np.fft.rfft(hamming_distance, NFFT))
    fft_trans = np.mean(fft) / np.max(fft)
    power_spec = np.around(fft[:NFFT//2], decimals=4)
    p_spec = ((1.0 / NFFT) * ((fft) ** 2))

    # computes the mel frequency cepstral coefficient of the sound signal
    mel_coeff = mel_coefficients(1000, 40, p_spec,num_cof)
    medain_power = np.median(power_spec)
    return mel_coeff,medain_power,fft_trans

def label_extraction(label_path):     #'/Users/mac/Desktop/heart_science/wav_label.txt'
    hs_label = txt_read(label_path)
    y_label = []
    for i in range(len(hs_label)):
        y_label.append(hs_label[i].split('\n')[0])
    return y_label

def txt_read(file_name):
    read_list = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            read_list.append(line)
    return read_list


def data_downsampling(file_path,save_path,sample_fs = 1000):
    data,sr = librosa.load(file_path)
    data_resample = librosa.resample(data,sr,sample_fs)
    data_sample = list(data_resample)

    with open(save_path,'w+') as f:
        for i in range(len(data_sample)):
            f.writelines(str(data_sample[i]))
            f.writelines('\n')


def contents_gain(contents_path):
    path_list = []
    print(len(os.listdir(contents_path)))
    for each_file in os.listdir(contents_path):
        path_list.append(os.path.join(contents_path,each_file))
    return path_list

def counter(input_list):          #找出心音变换时候的index
    now_list = np.diff(input_list)
    tran_index = []
    for i in range(len(now_list)):
        if now_list[i] != 0:
            tran_index.append(i+1)
    return tran_index

def element_div(list_1,list_2):                  #实现元素之间相除，用来计算s1/systole,s2/diastole
    result_list = []
    assert len(list_1) == len(list_2)
    for i in range(len(list_1)):
        result_list.append(np.round(list_1[i]/list_2[i],decimals=4))
    return result_list

def del_zero_element(mfcc_list,num_mel,num_zero):        #去除0的部分
    data_length = len(mfcc_list[0])
    for i in range(num_mel):
        mfcc_list[i] = mfcc_list[i][data_length - num_zero :]
    return mfcc_list

def different(list):
    A = []
    for i in range(2,len(list)-2):
        A.append(np.abs(np.round((-2*list[i-2]-list[i-1]+list[i+1]+2*list[i+2]),decimals=0))/np.sqrt(10))
        #A.append(np.round(list[i+1] - list[i],decimals=0))
    return A

def cal_diff(mfcc1,mfcc2):
    diff0 = []
    diff1 = []
    diff2 = []
    diff3 = []
    for i in range(12):
        buffer = []
        for j in range(len(mfcc1[i])):
            buffer.append(mfcc1[i][j])
            buffer.append(mfcc2[i][j])
            #buffer.append(mfcc3[i][j])
            #buffer.append(mfcc4[i][j])
        diff0.append(different(buffer))


    for i in range(12):
        diff1.append(different(diff0[i]))


    for m in range(12):
        for n in range(2):
            number_mean = []
            k = int(len(diff0[m])/2) - 1
            while k >= 0:
                if n == 0:
                    number_mean.append(diff0[m][2*k])
                elif n == 1:
                    number_mean.append(diff0[m][2*k+1])
                elif n == 2:
                    number_mean.append(diff0[m][4*k+2])
                else:
                    number_mean.append(diff0[m][4*k+3])
                k -= 1
            diff2.append(np.round(np.mean(number_mean),decimals=0))


    for m in range(12):
        for n in range(2):
            number_mean = []
            k = int(len(diff1[m])/2) - 1
            while k >= 0:
                if n == 0:
                    number_mean.append(diff1[m][2*k])
                elif n == 1:
                    number_mean.append(diff1[m][2*k+1])
                elif n == 2:
                    number_mean.append(diff1[m][4*k+2])
                else:
                    number_mean.append(diff1[m][4*k+3])
                k -= 1
            diff3.append(np.round(np.mean(number_mean),decimals=0))
    '''
    for m in range(12):
        diff2.append(np.round(np.mean(diff0[m])))
    for m in range(12):
        diff3.append(np.round(np.mean(diff1[m])))
    '''
    return diff2,diff3




def exteraction_feature(hs_amps_path,num_feature):
    wav_list = txt_read('/Users/mac/Desktop/heart_science/wav_path.txt')
    label_list = label_extraction('/Users/mac/Desktop/heart_science/wav_label.txt')
    data_feature = np.zeros((len(wav_list),num_feature)).tolist()
    contentes_amps = []
    contentes_state = []
    data_number = []

    hs_amps = []
    hs_state = []
    if hs_amps_path is None:
        for idx in range(len(wav_list)):
            save_path = '/home/deep/heart_science/hs_amps/undersampling_' + str(idx+1) + '.txt'
            data_downsampling(str(wav_list[idx].split('\n')[0]),save_path,sample_fs=1000)
            print('已完成第%d个下采样'%idx)
        print('下采样数据已经准备完毕')
    else:
        print('下采样数据已经准备完毕')


    for i in range(1,3240):
        contentes_amps.append('/Users/mac/Desktop/heart_science/hs_amps/undersampling_' + str(i) + '.txt')
    for i in range(1,3240):
        contentes_state.append('/Users/mac/Desktop/heart_science/hs_segment/wav_segment' + str(i) + '.txt')

    for i in range(len(contentes_amps)):
        data_number.append(i)
        with open(contentes_amps[i],'r+') as f:
            hs_amps.append(f.read())

    for i in range(len(contentes_state)):
        with open(contentes_state[i],'r+') as f:
            hs_state.append(f.read())

    for i,state,amp in zip(data_number,hs_state,hs_amps):
        feature_label = label_list[i]
        indivadul_list = []
        s1_skew = []
        fft_trans_list1 = []
        fft_trans_list2 = []
        feature_list1 = []
        s2_skew = []
        systole_skew = []
        diastole_skew = []

        s1_kurtosis = []
        s2_kurtosis = []
        systole_kurtosis = []
        diastole_kurtosis = []
        print('正在计算第{0}个数据'.format(i))
        state = np.array(list(state.split(',')),dtype='float32')
        amp = list(amp.split('\n'))
        change_position = counter(state)
        buffer_list = np.zeros((4,math.ceil(len(change_position)/4))).tolist()              #存放每个阶段的时长
        amp_list = np.zeros((4,math.ceil(len(change_position)/4))).tolist()
        power_list = np.zeros((4,math.ceil(len(change_position)/4))).tolist()

        _mel_s1_list = np.zeros((12,math.ceil(len(change_position)/4))).tolist()
        _mel_systole_list = np.zeros((12,math.ceil(len(change_position)/4))).tolist()
        _mel_s2_list = np.zeros((12,math.ceil(len(change_position)/4))).tolist()
        _mel_diastole_list = np.zeros((12,math.ceil(len(change_position)/4))).tolist()



        index = 0
        for j in range(len(change_position) - 1):
            now_state = int(state[change_position[j]] - 1)
            now_length = int(change_position[j+1]-change_position[j])
            now_amps = amp[change_position[j]:change_position[j+1]]

            mel_coeff,median_power,fft_trans = frequency_features(np.array(now_amps,dtype='float32'),12)
            mel_coeff = list(np.round(mel_coeff,decimals=4))

            try:
                buffer_list[now_state][index] = now_length
                amp_list[now_state][index] = now_amps
                power_list[now_state][index] = median_power                     #计算功率



                if now_state == 0:                                              #计算峰度以及偏斜度
                    s1_skew.append(skew(np.array(now_amps,dtype='float32')))
                    s1_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(12):
                        _mel_s1_list[cnt].append(mel_coeff[cnt])           #goto
                elif now_state == 1:
                    systole_skew.append(skew(np.array(now_amps,dtype='float32')))
                    systole_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(12):
                        _mel_systole_list[cnt].append(mel_coeff[cnt])
                    fft_trans_list1.append(fft_trans)
                elif now_state == 2:
                    s2_skew.append(skew(np.array(now_amps,dtype='float32')))
                    s2_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(12):
                        _mel_s2_list[cnt].append(mel_coeff[cnt])
                elif now_state == 3:
                    diastole_skew.append(skew(np.array(now_amps,dtype='float32')))
                    diastole_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(12):
                        _mel_diastole_list[cnt].append(mel_coeff[cnt])
                    fft_trans_list2.append(fft_trans)
            except IndexError:
                break

            if (j+1)%4 == 0:
                index += 1
            #将1234 替换成 0123
        indivadul_list.append(len(buffer_list[0]))
        indivadul_list.append(len(buffer_list[1]))
        indivadul_list.append(len(buffer_list[2]))
        indivadul_list.append(len(buffer_list[3]))
        num_cycle = min(indivadul_list) - 1

        s1_list = buffer_list[0][:num_cycle]
        systole_list = buffer_list[1][:num_cycle]
        s2_list = buffer_list[2][:num_cycle]
        diastole_list = buffer_list[3][:num_cycle]

        s1_power = power_list[0][:num_cycle]
        systole_power = power_list[1][:num_cycle]
        s2_power = power_list[2][:num_cycle]
        diastole_power = power_list[3][:num_cycle]

        mel_s1_list = del_zero_element(_mel_s1_list,12,num_cycle)
        mel_systole_list = del_zero_element(_mel_systole_list,12,num_cycle)
        mel_s2_list = del_zero_element(_mel_s2_list,12,num_cycle)
        mel_diastole_list = del_zero_element(_mel_diastole_list,12,num_cycle)




        mean_s1_power = np.round(np.mean(s1_power),decimals=4)
        std_s1_power = np.round(np.std(s1_power),decimals=4)
        mean_systole_power = np.round(np.mean(systole_power),decimals=4)
        std_systole_power = np.round(np.std(systole_power),decimals=4)
        mean_s2_power = np.round(np.mean(s2_power),decimals=4)
        std_s2_power = np.round(np.std(s2_power),decimals=4)
        mean_diastole_power = np.round(np.mean(diastole_power),decimals=4)
        std_diastole_power = np.round(np.std(diastole_power),decimals=4)



        mean_s1_time = np.round(np.mean(s1_list),decimals=4)           #计算time，单位ms
        std_s1_time = np.round(np.std(s1_list),decimals=4)
        mean_systole_time = np.round(np.mean(systole_list),decimals=4)
        std_systole_time = np.round(np.std(systole_list),decimals=4)
        mean_s2_time = np.round(np.mean(s2_list),decimals=4)
        std_s2_time = np.round(np.std(s2_list),decimals=4)
        mean_diastole_time = np.round(np.mean(diastole_list),decimals=4)
        std_diastole_time = np.round(np.std(diastole_list),decimals=4)

        mean_ratio_s1_systole = np.round(mean_s1_time/mean_systole_time,decimals=4)        #s1/收缩期（mean）
        mean_ratio_s2_diastole = np.round(mean_s2_time/mean_diastole_time,decimals=4)      #s2/舒张期

        std_ratio_s1_systole = np.round(np.std(element_div(s1_list,systole_list)),decimals=4)           #s1/收缩期（std）
        std_ratio_s2_diastole = np.round(np.std(element_div(s2_list,diastole_list)),decimals=4)         #s2/舒张期

        mean_s1_skew = np.round(np.mean(s1_skew),decimals=4)#
        std_s1_skew = np.round(np.std(s1_skew),decimals=4)#
        mean_systole_skew = np.round(np.mean(systole_skew),decimals=4)#
        std_systole_skew = np.round(np.std(systole_skew),decimals=4)#
        mean_s2_skew = np.round(np.mean(s2_skew),decimals=4)
        std_s2_skew = np.round(np.std(s2_skew),decimals=4)
        mean_diastole_skew = np.round(np.mean(diastole_skew),decimals=4) #
        std_diastole_skew = np.round(np.std(diastole_skew),decimals=4)   #

        mean_s1_kurtosis = np.round(np.mean(s1_kurtosis), decimals=4)
        std_s1_kurtosis = np.round(np.std(s1_kurtosis), decimals=4)
        mean_systole_kurtosis = np.round(np.mean(systole_kurtosis), decimals=4)
        std_systole_kurtosis = np.round(np.std(systole_kurtosis), decimals=4)
        mean_s2_kurtosis = np.round(np.mean(s2_kurtosis), decimals=4)
        std_s2_kurtosis = np.round(np.std(s2_kurtosis), decimals=4)
        mean_diastole_kurtosis = np.round(np.mean(diastole_kurtosis), decimals=4)
        std_diastole_kurtosis = np.round(np.std(diastole_kurtosis), decimals=4)

        diff_cof1, diff_cof2 = cal_diff(mel_diastole_list, mel_systole_list)


        mfcc_s1_1 = np.round(np.mean(mel_s1_list[0]))
        mfcc_s1_2 = np.round(np.mean(mel_s1_list[1]))
        mfcc_s1_3 = np.round(np.mean(mel_s1_list[2]))
        mfcc_s1_4 = np.round(np.mean(mel_s1_list[3]))
        mfcc_s1_5 = np.round(np.mean(mel_s1_list[4]))
        mfcc_s1_6 = np.round(np.mean(mel_s1_list[5]))
        mfcc_s1_7 = np.round(np.mean(mel_s1_list[6]))
        mfcc_s1_8 = np.round(np.mean(mel_s1_list[7]))
        mfcc_s1_9 = np.round(np.mean(mel_s1_list[8]))
        mfcc_s1_10 = np.round(np.mean(mel_s1_list[9]))
        mfcc_s1_11 = np.round(np.mean(mel_s1_list[10]))
        mfcc_s1_12 = np.round(np.mean(mel_s1_list[11]))

        mfcc_systole_1 = np.round(np.mean(mel_systole_list[0]))
        mfcc_systole_2 = np.round(np.mean(mel_systole_list[1]))
        mfcc_systole_3 = np.round(np.mean(mel_systole_list[2]))
        mfcc_systole_4 = np.round(np.mean(mel_systole_list[3]))
        mfcc_systole_5 = np.round(np.mean(mel_systole_list[4]))
        mfcc_systole_6 = np.round(np.mean(mel_systole_list[5]))
        mfcc_systole_7 = np.round(np.mean(mel_systole_list[6]))
        mfcc_systole_8 = np.round(np.mean(mel_systole_list[7]))
        mfcc_systole_9 = np.round(np.mean(mel_systole_list[8]))
        mfcc_systole_10 = np.round(np.mean(mel_systole_list[9]))
        mfcc_systole_11 = np.round(np.mean(mel_systole_list[10]))
        mfcc_systole_12 = np.round(np.mean(mel_systole_list[11]))

        mfcc_s2_1 = np.round(np.mean(mel_s2_list[0]))
        mfcc_s2_2 = np.round(np.mean(mel_s2_list[1]))
        mfcc_s2_3 = np.round(np.mean(mel_s2_list[2]))
        mfcc_s2_4 = np.round(np.mean(mel_s2_list[3]))
        mfcc_s2_5 = np.round(np.mean(mel_s2_list[4]))
        mfcc_s2_6 = np.round(np.mean(mel_s2_list[5]))
        mfcc_s2_7 = np.round(np.mean(mel_s2_list[6]))
        mfcc_s2_8 = np.round(np.mean(mel_s2_list[7]))
        mfcc_s2_9 = np.round(np.mean(mel_s2_list[8]))
        mfcc_s2_10 = np.round(np.mean(mel_s2_list[9]))
        mfcc_s2_11 = np.round(np.mean(mel_s2_list[10]))
        mfcc_s2_12 = np.round(np.mean(mel_s2_list[11]))

        mfcc_diastole_1 = np.round(np.mean(mel_diastole_list[0]))
        mfcc_diastole_2 = np.round(np.mean(mel_diastole_list[1]))
        mfcc_diastole_3 = np.round(np.mean(mel_diastole_list[2]))
        mfcc_diastole_4 = np.round(np.mean(mel_diastole_list[3]))
        mfcc_diastole_5 = np.round(np.mean(mel_diastole_list[4]))
        mfcc_diastole_6 = np.round(np.mean(mel_diastole_list[5]))
        mfcc_diastole_7 = np.round(np.mean(mel_diastole_list[6]))
        mfcc_diastole_8 = np.round(np.mean(mel_diastole_list[7]))
        mfcc_diastole_9 = np.round(np.mean(mel_diastole_list[8]))
        mfcc_diastole_10 = np.round(np.mean(mel_diastole_list[9]))
        mfcc_diastole_11 = np.round(np.mean(mel_diastole_list[10]))
        mfcc_diastole_12 = np.round(np.mean(mel_diastole_list[11]))

        feature_list = [mean_s1_time,std_s1_time,mean_systole_time,std_systole_time,\
                        mean_s2_time,std_s2_time,mean_diastole_time,std_diastole_time,\
                        mean_ratio_s1_systole,std_ratio_s1_systole,mean_ratio_s2_diastole,std_ratio_s2_diastole,\
                        mean_s1_skew,std_s1_skew,mean_systole_skew,std_systole_skew,mean_s2_skew,std_s2_skew,\
                        mean_diastole_skew,std_diastole_skew,mean_s1_kurtosis,std_s1_kurtosis,mean_systole_kurtosis,\
                        std_systole_kurtosis,mean_s2_kurtosis,std_s2_kurtosis,mean_diastole_kurtosis,std_diastole_kurtosis, \
                        mean_s1_power,std_s1_power,mean_systole_power,std_systole_power,mean_s2_power,std_s2_power, \
                        mean_diastole_power,std_diastole_power,mfcc_s1_1,mfcc_s1_2,mfcc_s1_3,mfcc_s1_4,mfcc_s1_5,\
                        mfcc_s1_6,mfcc_s1_7,mfcc_s1_8,mfcc_s1_9,mfcc_s1_10,mfcc_s1_11,mfcc_s1_12, \
                        mfcc_systole_1,mfcc_systole_2,mfcc_systole_3,mfcc_systole_4,mfcc_systole_5,mfcc_systole_6,\
                        mfcc_systole_7,mfcc_systole_8,mfcc_systole_9,mfcc_systole_10,mfcc_systole_11,mfcc_systole_12,\
                        mfcc_s2_1,mfcc_s2_2,mfcc_s2_3,mfcc_s2_4,mfcc_s2_5,mfcc_s2_6,mfcc_s2_7,mfcc_s2_8,mfcc_s2_9, \
                        mfcc_s2_10,mfcc_s2_11,mfcc_s2_12, \
                        mfcc_diastole_1,mfcc_diastole_2,mfcc_diastole_3,mfcc_diastole_4,mfcc_diastole_5,mfcc_diastole_6, \
                        mfcc_diastole_7,mfcc_diastole_8,mfcc_diastole_9,mfcc_diastole_10,mfcc_diastole_11,mfcc_diastole_12]


        for idx in range(len(feature_list)):
            data_feature[i][idx] = feature_list[idx]
        data_feature[i].append(np.mean(fft_trans_list1))
        data_feature[i].append(np.mean(fft_trans_list2))
        data_feature[i] = data_feature[i] + diff_cof1 + diff_cof2
        data_feature[i].append(feature_label)


    return data_feature


if __name__ ==  "__main__":
    '''
    feature_name = ['mean_s1_time','std_s1_time','mean_systole_time','std_systole_time',\
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
                        'mfcc_diastole_7','mfcc_diastole_8','mfcc_diastole_9','mfcc_diastole_10','mfcc_diastole_11','mfcc_diastole_12','feature_label']
    '''

    extracted_feature = exteraction_feature('/home/deep/heart_science/hs_amps.txt',84)
    df_feat_ext = pd.DataFrame(extracted_feature)

    out_file = '/Users/mac/Desktop/heart_science/data_feature0.csv'
    try:
        df_feat_ext.to_csv(out_file, index=False)
    except Exception:
        print("Output path does not exist")









































