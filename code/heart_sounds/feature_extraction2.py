import numpy as np
import csv
import os
import wave
import librosa
import math
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.fftpack import dct
from utils import undersampling
from stacking_method import model_training_stack
import signal
from python_speech_features import mfcc
import pywt

NFFT = 256
NFFT1 = 256
n_maj = 0.25
n_min = 1.0
epochs = 10

len_frame = 256
frame_mov = 80

def cal_skew(data_sequence):
    frame_data = enframe(data_sequence, len_frame, frame_mov,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    skew_list = []
    for w in range(frame_data.shape[0]):
        skew_list.append(skew(frame_data[w]))
    return np.mean(skew_list)

def cal_kurtosis(data_sequence):
    frame_data = enframe(data_sequence, len_frame, frame_mov,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    kurtosis_list = []
    for w in range(frame_data.shape[0]):
        kurtosis_list.append(skew(frame_data[w]))
    return np.mean(kurtosis_list)


def cal_energy(data_sequence):
    frame_data = enframe(data_sequence, len_frame, frame_mov,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    sum_list = []
    for w in range(frame_data.shape[0]):
        sum = 0
        for i in range(len(frame_data[w])):
            sum += pow(frame_data[w][i],2)
        sum_list.append(sum)
    return np.mean(sum_list)

def zero_pass(data_sequence):
    frame_data = enframe(data_sequence, len_frame, frame_mov,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    count_list = []
    for w in range(frame_data.shape[0]):
        count = 0
        for i in range(len(frame_data) - 1):
            if frame_data[w][i] * frame_data[w][i+1] < 0:
                count+=1
            count_list.append(count)
    return np.mean(count_list)

def cal_mean(wav_in):
    last_list = []
    for i in range(20):
        buf_list = []
        for j in range(len(wav_in)):
            buf_list.append(wav_in[j][i])
        last_list.append(np.mean(buf_list))
    return last_list

def enframe(wave_data, nw, inc, winfunc):
    wlen = len(wave_data)  # 信号总长度
    if wlen <= nw:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * wlen - nw + inc) / inc))
    pad_length = int((nf - 1) * inc + nw)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - wlen,))
    pad_signal = np.concatenate((wave_data, zeros))
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    win = np.tile(winfunc, (nf, 1))
    return frames * win

def mel_coefficients(sample_rate, nfilt, pow_frames,num_cof):
    low_freq_mel = 0
    num_mel_coeff = num_cof
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2.0) / 700.0))  # Convert Hz to Mel

    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT1 + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT1 / 2 + 1))))
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

def total_frequency(data_sequence):
    tol_mel = []
    frame_data = enframe(data_sequence, len_frame, frame_mov,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    for i in range(frame_data.shape[0]):
        pro_seq = frame_data[i,:]
        fft = np.absolute(np.fft.rfft(pro_seq, NFFT))
        p_spec = ((1.0 / NFFT) * ((fft) ** 2))
        mel_coeff = mel_coefficients(1000, 40, p_spec, 20)
        tol_mel.append(list(mel_coeff))
    return cal_mean(tol_mel)

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

def exteraction_feature(hs_amps_path,num_feature):
    wav_list = txt_read('/home/deep/heart_science/wav_path.txt')
    label_list = label_extraction('/home/deep/heart_science/wav_label.txt')
    data_feature = np.zeros((len(wav_list), num_feature)).tolist()
    contentes_amps = []
    contentes_state = []
    data_number = []

    hs_amps = []
    hs_state = []
    if hs_amps_path is None:
        for idx in range(len(wav_list)):
            save_path = '/home/deep/heart_science/hs_amps/undersampling_' + str(idx + 1) + '.txt'
            data_downsampling(str(wav_list[idx].split('\n')[0]), save_path, sample_fs=1000)
            print('已完成第%d个下采样' % idx)
        print('下采样数据已经准备完毕')
    else:
        print('下采样数据已经准备完毕')

    for i in range(1, 3240):
        contentes_amps.append('/home/deep/heart_science/hs_amps/undersampling_' + str(i) + '.txt')
    for i in range(1, 3240):
        contentes_state.append('/home/deep/heart_science/hs_segment/wav_segment' + str(i) + '.txt')

    for i in range(len(contentes_amps)):
        data_number.append(i)
        with open(contentes_amps[i], 'r+') as f:
            hs_amps.append(f.read())

    for i in range(len(contentes_state)):
        with open(contentes_state[i], 'r+') as f:
            hs_state.append(f.read())

    for i,state,amp in zip(data_number,hs_state,hs_amps):
        try:
            indivadul_list = []
            feature_label = label_list[i]

            list1 = []
            list2 = []
            list3 = []
            list4 = []


            print('正在计算第{0}个数据'.format(i))
            state = np.array(list(state.split(',')),dtype='float32')
            amp = list(amp.split('\n'))
            change_position = counter(state)
            buffer_list = np.zeros((4,math.ceil(len(change_position)/4))).tolist()              #存放每个阶段的时长
            amp_list = np.zeros((4,math.ceil(len(change_position)/4))).tolist()




            index = 0
            for j in range(len(change_position) - 1):
                now_state = int(state[change_position[j]] - 1)
                now_length = int(change_position[j+1]-change_position[j])
                now_amps = amp[change_position[j]:change_position[j+1]]


                try:
                    buffer_list[now_state][index] = now_length
                    amp_list[now_state][index] = now_amps

                    if now_state == 0:                                              #计算峰度以及偏斜度
                        list1.extend(now_amps)
                    elif now_state == 1:
                        list2.extend(now_amps)
                    elif now_state == 2:
                        list3.extend(now_amps)
                    elif now_state == 3:
                        list4.extend(now_amps)
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


            list1 = np.array(list1,dtype='float32')
            list2 = np.array(list2,dtype='float32')
            list3 = np.array(list3,dtype='float32')
            list4 = np.array(list4,dtype='float32')

            _mel_s1_list = total_frequency(list1)  # mfcc
            s1_zero = zero_pass(list1)  # 短时过0率
            s1_energy = cal_energy(list1)  # 短时能量
            s1_skew = cal_skew(list1)
            s1_kurtosis = cal_kurtosis(list1)

            _mel_systole_list = total_frequency(list2)  # mfcc
            systole_zero = zero_pass(list2)  # 短时过0率
            systole_energy = cal_energy(list2)  # 短时能量
            systole_skew = cal_skew(list2)
            systole_kurtosis = cal_kurtosis(list2)

            _mel_s2_list = total_frequency(list3)  # mfcc
            s2_zero = zero_pass(list3)  # 短时过0率
            s2_energy = cal_energy(list3)  # 短时能量
            s2_skew = cal_skew(list3)
            s2_kurtosis = cal_kurtosis(list3)

            _mel_diastole_list = total_frequency(list4)  # mfcc
            diastole_zero = zero_pass(list4)  # 短时过0率
            diastole_energy = cal_energy(list4)  # 短时能量
            diastole_skew = cal_skew(list4)
            diastole_kurtosis = cal_kurtosis(list4)

            feature_list = [np.mean(_mel_s1_list[0]),np.mean(_mel_s1_list[1]),np.mean(_mel_s1_list[2]),np.mean(_mel_s1_list[3]),
                            np.mean(_mel_s1_list[4]),np.mean(_mel_s1_list[5]),np.mean(_mel_s1_list[6]),np.mean(_mel_s1_list[7]),
                            np.mean(_mel_s1_list[8]),np.mean(_mel_s1_list[9]),np.mean(_mel_s1_list[10]),np.mean(_mel_s1_list[11]),
                            np.mean(_mel_systole_list[0]),np.mean(_mel_systole_list[1]),np.mean(_mel_systole_list[2]),np.mean(_mel_systole_list[3]),
                            np.mean(_mel_systole_list[4]),np.mean(_mel_systole_list[5]),np.mean(_mel_systole_list[6]),np.mean(_mel_systole_list[7]),
                            np.mean(_mel_systole_list[8]),np.mean(_mel_systole_list[9]),np.mean(_mel_systole_list[10]),np.mean(_mel_systole_list[11]),
                            np.mean(_mel_s2_list[0]),np.mean(_mel_s2_list[1]),np.mean(_mel_s2_list[2]),np.mean(_mel_s2_list[3]),
                            np.mean(_mel_s2_list[4]),np.mean(_mel_s2_list[5]),np.mean(_mel_s2_list[6]),np.mean(_mel_s2_list[7]),
                            np.mean(_mel_s2_list[8]),np.mean(_mel_s2_list[9]),np.mean(_mel_s2_list[10]),np.mean(_mel_s2_list[11]),
                            np.mean(_mel_diastole_list[0]),np.mean(_mel_diastole_list[1]),np.mean(_mel_diastole_list[2]),np.mean(_mel_diastole_list[3]),
                            np.mean(_mel_diastole_list[4]),np.mean(_mel_diastole_list[5]),np.mean(_mel_diastole_list[6]),np.mean(_mel_diastole_list[7]),
                            np.mean(_mel_diastole_list[8]),np.mean(_mel_diastole_list[9]),np.mean(_mel_diastole_list[10]),np.mean(_mel_diastole_list[11]),
                            np.mean(s1_energy),np.mean(s1_zero),np.mean(s1_kurtosis),np.mean(s1_skew),
                            np.mean(systole_energy),np.mean(systole_zero),np.mean(systole_kurtosis),np.mean(systole_skew),
                            np.mean(s2_energy), np.mean(s2_zero), np.mean(s2_kurtosis), np.mean(s2_skew),
                            np.mean(diastole_energy), np.mean(diastole_zero), np.mean(diastole_kurtosis), np.mean(diastole_skew)]
            for idx in range(len(feature_list)):
                data_feature[i][idx] = feature_list[idx]
            data_feature[i].extend(_mel_systole_list[11:20])
            data_feature[i].extend(_mel_diastole_list[11:20])
            data_feature[i].append(feature_label)
        except:
            pass

    return data_feature


if __name__ ==  "__main__":

    feature_name = ['mfcc_s1_1','mfcc_s1_2','mfcc_s1_3','mfcc_s1_4','mfcc_s1_5','mfcc_s1_6','mfcc_s1_7','mfcc_s1_8',
                    'mfcc_s1_9','mfcc_s1_10','mfcc_s1_11','mfcc_s1_12','mfcc_systole_1','mfcc_systole_2','mfcc_systole_3',
                    'mfcc_systole_4','mfcc_systole_5','mfcc_systole_6','mfcc_systole_7','mfcc_systole_8','mfcc_systole_9',
                    'mfcc_systole_10','mfcc_systole_11','mfcc_systole_12','mfcc_s2_1','mfcc_s2_2','mfcc_s2_3','mfcc_s2_4',
                    'mfcc_s2_5','mfcc_s2_6','mfcc_s2_7','mfcc_s2_8','mfcc_s2_9','mfcc_s2_10','mfcc_s2_11','mfcc_s2_12',
                    'mfcc_diastole_1','mfcc_diastole_2','mfcc_diastole_3','mfcc_diastole_4','mfcc_diastole_5','mfcc_diastole_6',
                    'mfcc_diastole_7','mfcc_diastole_8','mfcc_diastole_9','mfcc_diastole_10','mfcc_diastole_11','mfcc_diastole_12',
                    's1_energy','s1_zero','s1_kurtosis','s1_skew','systole_energy','systole_zero','systole_kurtosis','systole_skew',
                    's2_energy','s2_zero','s2_kurtosis','s2_skew','diastole_energy','diastole_zero','diastole_kurtosis','diastole_skew',
                    'label']
    extracted_feature = exteraction_feature('/home/deep/heart_science/hs_amps.txt',64)
    df_feat_ext = pd.DataFrame(extracted_feature)

    out_file = '/home/deep/heart_science/data_feature1.csv'
    try:
        df_feat_ext.to_csv(out_file, index=False)
    except Exception:
        print("Output path does not exist")