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


def mel_dif(list_in):
    dif_list1 = []
    dif_list2 = []
    buffer = []
    for j in range(12):
        number_list0 = []
        for i in range(len(list_in)):
            number_list0.append(list_in[i][j])
        A,B = different(number_list0)
        dif_list1.append(A)
        buffer.append(B)
    for j in range(12):
        number_list1 = []
        for i in range(len(buffer[0])):
            number_list1.append(buffer[j][i])
        C,_ = different(number_list1)
        dif_list2.append(C)

    return dif_list1,dif_list2
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

def frequency_features(data_sequence,num_cof):

    # computes the power spectrum of the signal
    hamming_distance = data_sequence*np.hamming(len(data_sequence))
    fft = np.absolute(np.fft.rfft(hamming_distance, NFFT))
    fft_trans = np.mean(fft)/np.max(fft)
    power_spec = np.around(fft[:NFFT//2], decimals=4)
    p_spec = ((1.0 / NFFT) * ((fft) ** 2))


    # computes the mel frequency cepstral coefficient of the sound signal
    mel_coeff = mel_coefficients(1000, 40, p_spec,num_cof)
    medain_power = np.median(power_spec)
    return mel_coeff,medain_power,fft_trans

def total_frequency(data_sequence):
    tol_mel = []
    frame_data = enframe(data_sequence, 256, 80,np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (NFFT - 1)) for n in range(NFFT)]))
    for i in range(frame_data.shape[0]):
        pro_seq = frame_data[i,:]
        fft = np.absolute(np.fft.rfft(pro_seq, NFFT))
        p_spec = ((1.0 / NFFT) * ((fft) ** 2))
        mel_coeff = mel_coefficients(1000, 40, p_spec, 12)
        tol_mel.append(list(mel_coeff))
    return mel_dif(tol_mel)

'''
def frequency_features(data_sequence,num_cof):

    # computes the power spectrum of the signal
    hamming_distance = data_sequence*np.hamming(len(data_sequence))
    A2,_,_,_,_ = pywt.wavedec(hamming_distance,'db4',mode='sym',level=4)
    wl0 = np.absolute(np.fft.rfft(A2, NFFT))
    wdt = list(wl0)[:129]
    wdt = np.array(wdt,dtype='float32')
    power_spec = np.around(wdt[:NFFT//2], decimals=4)
    p_spec = ((1.0 / NFFT) * ((wdt) ** 2))


    # computes the mel frequency cepstral coefficient of the sound signal
    mel_coeff = mel_coefficients(1000, 40, p_spec,num_cof)
    medain_power = np.median(power_spec)
    return mel_coeff,medain_power
'''

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
    return np.round(np.mean(A),decimals=0),A
'''
def cal_diff(mfcc1,mfcc2,num_get):
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

    for m in range(num_get):
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


    for m in range(num_get):
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

    for m in range(num_get):
        diff2.append(np.round(np.mean(diff0[m])))
    for m in range(num_get):
        diff3.append(np.round(np.mean(diff1[m])))

    return diff2,diff3

'''




def exteraction_feature(hs_amps_path,num_cof,num_get):
    wav_list = txt_read('/home/deep/heart_science/wav_path.txt')
    label_list = label_extraction('/home/deep/heart_science/wav_label.txt')
    #data_feature0 = np.zeros((len(wav_list),num_cof*2)).tolist()
    #data_feature1 = np.zeros((len(wav_list),(num_cof*2) + 12)).tolist()
    #data_feature2 = np.zeros((len(wav_list), (num_cof * 2) + 24)).tolist()
    data_feature0 = np.zeros((len(wav_list), 24)).tolist()
    data_feature1 = np.zeros((len(wav_list), 36)).tolist()
    data_feature2 = np.zeros((len(wav_list), 48)).tolist()
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
        contentes_amps.append('/home/deep/heart_science/hs_amps/undersampling_' + str(i) + '.txt')
    for i in range(1,3240):
        contentes_state.append('/home/deep/heart_science/hs_segment/wav_segment' + str(i) + '.txt')

    for i in range(len(contentes_amps)):
        data_number.append(i)
        with open(contentes_amps[i],'r+') as f:
            hs_amps.append(f.read())

    for i in range(len(contentes_state)):
        with open(contentes_state[i],'r+') as f:
            hs_state.append(f.read())

    for i,state,amp in zip(data_number,hs_state,hs_amps):
        indivadul_list = []
        feature_label = label_list[i]
        s1_skew = []
        feature_list1 = []
        s2_skew = []
        systole_skew = []
        diastole_skew = []
        enhance_list1 = []
        enhance_list2 = []
        fft_trans_list1 = []
        fft_trans_list2 = []

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

        _mel_s1_list = np.zeros((num_cof,math.ceil(len(change_position)/4))).tolist()
        _mel_systole_list = np.zeros((num_cof,math.ceil(len(change_position)/4))).tolist()
        _mel_s2_list = np.zeros((num_cof,math.ceil(len(change_position)/4))).tolist()
        _mel_diastole_list = np.zeros((num_cof,math.ceil(len(change_position)/4))).tolist()




        index = 0
        for j in range(len(change_position) - 1):
            now_state = int(state[change_position[j]] - 1)
            now_length = int(change_position[j+1]-change_position[j])
            now_amps = amp[change_position[j]:change_position[j+1]]
            mel_coeff,median_power,fft_trans = frequency_features(np.array(now_amps,dtype='float32'),num_cof)
            mel_coeff = list(np.round(mel_coeff,decimals=4))


            try:
                buffer_list[now_state][index] = now_length
                amp_list[now_state][index] = now_amps
                power_list[now_state][index] = median_power                     #计算功率




                if now_state == 0:                                              #计算峰度以及偏斜度
                    s1_skew.append(skew(np.array(now_amps,dtype='float32')))
                    s1_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(num_cof):
                        _mel_s1_list[cnt].append(mel_coeff[cnt])           #goto
                    enhance_list1.extend(now_amps)
                elif now_state == 1:
                    systole_skew.append(skew(np.array(now_amps,dtype='float32')))
                    systole_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(num_cof):
                        _mel_systole_list[cnt].append(mel_coeff[cnt])
                    fft_trans_list1.append(fft_trans)
                    enhance_list1.extend(now_amps)
                elif now_state == 2:
                    s2_skew.append(skew(np.array(now_amps,dtype='float32')))
                    s2_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(num_cof):
                        _mel_s2_list[cnt].append(mel_coeff[cnt])
                    enhance_list1.extend(now_amps)
                elif now_state == 3:
                    diastole_skew.append(skew(np.array(now_amps,dtype='float32')))
                    diastole_kurtosis.append(kurtosis(np.array(now_amps,dtype='float32')))
                    for cnt in range(num_cof):
                        _mel_diastole_list[cnt].append(mel_coeff[cnt])
                    fft_trans_list2.append(fft_trans)
                    enhance_list1.extend(now_amps)
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
        '''
        one_cycle = amp_list[0][1] + amp_list[1][1] + amp_list[2][1] + amp_list[3][1]
        one_cycle = np.round(np.array(one_cycle,dtype="float32").tolist(),decimals=3)
        mfcc_cycle = mfcc(one_cycle,1000)
        print(len(mfcc_cycle))
        '''


        mel_s1_list = del_zero_element(_mel_s1_list,12,num_cycle)
        mel_systole_list = del_zero_element(_mel_systole_list,12,num_cycle)
        mel_s2_list = del_zero_element(_mel_s2_list,12,num_cycle)
        mel_diastole_list = del_zero_element(_mel_diastole_list,12,num_cycle)

        print(len(np.array(enhance_list1, dtype='float32')))
        enhance_mel1, enhance_mel2 = total_frequency(np.array(enhance_list1, dtype='float32'))
        #enhance_mel21, enhance_mel22 = total_frequency(np.array(enhance_list1, dtype='float32'))






        middle1 = []
        for time1 in mel_systole_list:
            feature_list1.append(np.round(np.mean(time1)))

        middle1 = []
        for time2 in mel_diastole_list:
            feature_list1.append(np.round(np.mean(time2)))







        '''
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
        mfcc_systole_13 = np.round(np.mean(mel_systole_list[12]))
        mfcc_systole_14 = np.round(np.mean(mel_systole_list[13]))
        mfcc_systole_15 = np.round(np.mean(mel_systole_list[14]))
        mfcc_systole_16 = np.round(np.mean(mel_systole_list[15]))
        mfcc_systole_17 = np.round(np.mean(mel_systole_list[16]))
        mfcc_systole_18 = np.round(np.mean(mel_systole_list[17]))
        mfcc_systole_19 = np.round(np.mean(mel_systole_list[18]))
        mfcc_systole_20 = np.round(np.mean(mel_systole_list[19]))
        mfcc_systole_21 = np.round(np.mean(mel_systole_list[20]))
        mfcc_systole_22 = np.round(np.mean(mel_systole_list[21]))
        mfcc_systole_23 = np.round(np.mean(mel_systole_list[22]))
        mfcc_systole_24 = np.round(np.mean(mel_systole_list[23]))
        mfcc_systole_25 = np.round(np.mean(mel_systole_list[24]))
        mfcc_systole_26 = np.round(np.mean(mel_systole_list[25]))
        mfcc_systole_27 = np.round(np.mean(mel_systole_list[26]))
        mfcc_systole_28 = np.round(np.mean(mel_systole_list[27]))
        mfcc_systole_29 = np.round(np.mean(mel_systole_list[28]))
        mfcc_systole_30 = np.round(np.mean(mel_systole_list[29]))
        mfcc_systole_31 = np.round(np.mean(mel_systole_list[30]))
        mfcc_systole_32 = np.round(np.mean(mel_systole_list[31]))
        mfcc_systole_33 = np.round(np.mean(mel_systole_list[32]))
        mfcc_systole_34 = np.round(np.mean(mel_systole_list[33]))
        mfcc_systole_35 = np.round(np.mean(mel_systole_list[34]))
        mfcc_systole_36 = np.round(np.mean(mel_systole_list[35]))
        mfcc_systole_37 = np.round(np.mean(mel_systole_list[36]))
        mfcc_systole_38 = np.round(np.mean(mel_systole_list[37]))
        mfcc_systole_39 = np.round(np.mean(mel_systole_list[38]))
        mfcc_systole_40 = np.round(np.mean(mel_systole_list[39]))
        mfcc_systole_41 = np.round(np.mean(mel_systole_list[40]))
        mfcc_systole_42 = np.round(np.mean(mel_systole_list[41]))
        mfcc_systole_43 = np.round(np.mean(mel_systole_list[42]))
        mfcc_systole_44 = np.round(np.mean(mel_systole_list[43]))
        mfcc_systole_45 = np.round(np.mean(mel_systole_list[44]))
        mfcc_systole_46 = np.round(np.mean(mel_systole_list[45]))
        mfcc_systole_47 = np.round(np.mean(mel_systole_list[46]))
        mfcc_systole_48 = np.round(np.mean(mel_systole_list[47]))
        '''

        '''
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
        mfcc_diastole_13 = np.round(np.mean(mel_diastole_list[12]))
        mfcc_diastole_14 = np.round(np.mean(mel_diastole_list[13]))
        mfcc_diastole_15 = np.round(np.mean(mel_diastole_list[14]))
        mfcc_diastole_16 = np.round(np.mean(mel_diastole_list[15]))
        mfcc_diastole_17 = np.round(np.mean(mel_diastole_list[16]))
        mfcc_diastole_18 = np.round(np.mean(mel_diastole_list[17]))
        mfcc_diastole_19 = np.round(np.mean(mel_diastole_list[18]))
        mfcc_diastole_20 = np.round(np.mean(mel_diastole_list[19]))
        mfcc_diastole_21 = np.round(np.mean(mel_diastole_list[20]))
        mfcc_diastole_22 = np.round(np.mean(mel_diastole_list[21]))
        mfcc_diastole_23 = np.round(np.mean(mel_diastole_list[22]))
        mfcc_diastole_24 = np.round(np.mean(mel_diastole_list[23]))
        mfcc_diastole_25 = np.round(np.mean(mel_diastole_list[24]))
        mfcc_diastole_26 = np.round(np.mean(mel_diastole_list[25]))
        mfcc_diastole_27 = np.round(np.mean(mel_diastole_list[26]))
        mfcc_diastole_28 = np.round(np.mean(mel_diastole_list[27]))
        mfcc_diastole_29 = np.round(np.mean(mel_diastole_list[28]))
        mfcc_diastole_30 = np.round(np.mean(mel_diastole_list[29]))
        mfcc_diastole_31 = np.round(np.mean(mel_diastole_list[30]))
        mfcc_diastole_32 = np.round(np.mean(mel_diastole_list[31]))
        mfcc_diastole_33 = np.round(np.mean(mel_diastole_list[32]))
        mfcc_diastole_34 = np.round(np.mean(mel_diastole_list[33]))
        mfcc_diastole_35 = np.round(np.mean(mel_diastole_list[34]))
        mfcc_diastole_36 = np.round(np.mean(mel_diastole_list[35]))
        mfcc_diastole_37 = np.round(np.mean(mel_diastole_list[36]))
        mfcc_diastole_38 = np.round(np.mean(mel_diastole_list[37]))
        mfcc_diastole_39 = np.round(np.mean(mel_diastole_list[38]))
        mfcc_diastole_40 = np.round(np.mean(mel_diastole_list[39]))
        mfcc_diastole_41 = np.round(np.mean(mel_diastole_list[40]))
        mfcc_diastole_42 = np.round(np.mean(mel_diastole_list[41]))
        mfcc_diastole_43 = np.round(np.mean(mel_diastole_list[42]))
        mfcc_diastole_44 = np.round(np.mean(mel_diastole_list[43]))
        mfcc_diastole_45 = np.round(np.mean(mel_diastole_list[44]))
        mfcc_diastole_46 = np.round(np.mean(mel_diastole_list[45]))
        mfcc_diastole_47 = np.round(np.mean(mel_diastole_list[46]))
        mfcc_diastole_48 = np.round(np.mean(mel_diastole_list[47]))

        feature_list = [
                        mfcc_systole_1,mfcc_systole_2,mfcc_systole_3,mfcc_systole_4,mfcc_systole_5,mfcc_systole_6,\
                        mfcc_systole_7,mfcc_systole_8,mfcc_systole_9,mfcc_systole_10,mfcc_systole_11,mfcc_systole_12, \
                        mfcc_systole_13, mfcc_systole_14, mfcc_systole_15, mfcc_systole_16, mfcc_systole_17, mfcc_systole_18, \
                        mfcc_systole_19, mfcc_systole_20, mfcc_systole_21, mfcc_systole_22, mfcc_systole_23,mfcc_systole_24, \
                        mfcc_systole_25, mfcc_systole_26, mfcc_systole_27, mfcc_systole_28, mfcc_systole_29, mfcc_systole_30, \
                        mfcc_systole_31, mfcc_systole_32, mfcc_systole_33, mfcc_systole_34, mfcc_systole_35, mfcc_systole_36, \
                        #mfcc_systole_37, mfcc_systole_38, mfcc_systole_39, mfcc_systole_40, mfcc_systole_41, mfcc_systole_42, \
                        #mfcc_systole_43, mfcc_systole_44, mfcc_systole_45, mfcc_systole_46, mfcc_systole_47, mfcc_systole_48, \
                        mfcc_diastole_1, mfcc_diastole_2, mfcc_diastole_3, mfcc_diastole_4, mfcc_diastole_5, mfcc_diastole_6, \
                        mfcc_diastole_7, mfcc_diastole_8, mfcc_diastole_9, mfcc_diastole_10, mfcc_diastole_11, mfcc_diastole_12, \
                        mfcc_diastole_13, mfcc_diastole_14, mfcc_diastole_15, mfcc_diastole_16, mfcc_diastole_17, mfcc_diastole_18, \
                        mfcc_diastole_19, mfcc_diastole_20, mfcc_diastole_21, mfcc_diastole_22, mfcc_diastole_23, mfcc_diastole_24, \
                        mfcc_diastole_25, mfcc_diastole_26, mfcc_diastole_27, mfcc_diastole_28, mfcc_diastole_29, mfcc_diastole_30, \
                        mfcc_diastole_31, mfcc_diastole_32, mfcc_diastole_33, mfcc_diastole_34, mfcc_diastole_35, mfcc_diastole_36, \
                        #mfcc_diastole_37, mfcc_diastole_38, mfcc_diastole_39, mfcc_diastole_40, mfcc_diastole_41, mfcc_diastole_42, \
                        #mfcc_diastole_43, mfcc_diastole_44, mfcc_diastole_45, mfcc_diastole_46, mfcc_diastole_47, mfcc_diastole_48, \
                        feature_label
            ]
    '''
        for idx in range(len(feature_list1)):
            data_feature0[i][idx] = feature_list1[idx]
        data_feature0[i].append(np.mean(fft_trans_list1))
        data_feature0[i].append(np.mean(fft_trans_list2))
        data_feature1[i] = data_feature0[i] + enhance_mel1
        data_feature2[i] = data_feature0[i] + enhance_mel1 + enhance_mel2

        data_feature0[i].append(feature_label)
        data_feature1[i].append(feature_label)
        data_feature2[i].append(feature_label)



    return data_feature0,data_feature1,data_feature2


if __name__ ==  "__main__":
    num_get = 12
    num_cof = 12

    while num_get < 13:
        extracted_feature0,extracted_feature1,extracted_feature2 = exteraction_feature('/home/deep/heart_science/hs_amps.txt',num_cof,num_get)
        df_feat_ext0 = pd.DataFrame(extracted_feature0)
        df_feat_ext1 = pd.DataFrame(extracted_feature1)
        df_feat_ext2 = pd.DataFrame(extracted_feature2)


        out_file0 = '/home/deep/heart_science/feature_enhance0/feature_enhance' + str(num_get) + '.csv'
        out_file1 = '/home/deep/heart_science/feature_enhance1/feature_enhance' + str(num_get) + '.csv'
        out_file2 = '/home/deep/heart_science/feature_enhance2/feature_enhance' + str(num_get) + '.csv'
        num_get += 1
        try:
            df_feat_ext0.to_csv(out_file0, index=False)
            df_feat_ext1.to_csv(out_file1, index=False)
            df_feat_ext2.to_csv(out_file2, index = False)
        except Exception:
            print("Output path does not exist")

    path = '/home/deep/heart_science/feature_enhance0'
    data_subset = os.listdir(path)

    for i in range(len(data_subset)):
        data_path0 = '/home/deep/heart_science/feature_enhance0/' + str(data_subset[i])
        data_path1 = '/home/deep/heart_science/feature_enhance1/' + str(data_subset[i])
        data_path2 = '/home/deep/heart_science/feature_enhance2/' + str(data_subset[i])

        train_data0 = pd.read_csv(data_path0)
        train_data1 = pd.read_csv(data_path1)
        train_data2 = pd.read_csv(data_path2)

        train_data0.dropna(inplace=True)
        train_data1.dropna(inplace=True)
        train_data2.dropna(inplace=True)

        feature_list0 = train_data0.columns.values.tolist()
        feature_list1 = train_data1.columns.values.tolist()
        feature_list2 = train_data2.columns.values.tolist()

        feature_label0 = feature_list0[-1]
        feature_label1 = feature_list1[-1]
        feature_label2 = feature_list2[-1]
        feature_list0.remove(feature_label0)
        feature_list1.remove(feature_label1)
        feature_list2.remove(feature_label2)

        X_0 = train_data0[feature_list0]
        Y_0 = train_data0[feature_label0]
        X_1 = train_data1[feature_list1]
        Y_1 = train_data1[feature_label1]
        X_2 = train_data2[feature_list2]
        Y_2 = train_data2[feature_label2]

        X0, Y0 = undersampling(X_0.values, Y_0.values, majority_class=-1, minority_class=1,
                             maj_proportion=n_maj, min_proportion=n_min)

        X1, Y1 = undersampling(X_1.values, Y_1.values, majority_class=-1, minority_class=1,
                               maj_proportion=n_maj, min_proportion=n_min)

        X2, Y2 = undersampling(X_2.values, Y_2.values, majority_class=-1, minority_class=1,
                               maj_proportion=n_maj, min_proportion=n_min)

        sensitivity0, recall0, fscore0, MACC0 = model_training_stack(X0, Y0, True, 'feature_label', n_maj, n_min)
        sensitivity1, recall1, fscore1, MACC1 = model_training_stack(X1, Y1, True, 'feature_label', n_maj, n_min)
        sensitivity2, recall2, fscore2, MACC2 = model_training_stack(X2, Y2, True, 'feature_label', n_maj, n_min)

        save_path0 = '/home/deep/heart_science/dicuss0.txt'
        save_path1 = '/home/deep/heart_science/dicuss1.txt'
        save_path2 = '/home/deep/heart_science/dicuss2.txt'

        with open(save_path0, 'a+') as f:
            f.write('----------------------------------------------------------------')
            f.write('\n')
            f.write('num mfcc is : ')
            f.write(data_subset[i])
            f.write('\n')
            f.write('sensitivity is ')
            f.write(str(sensitivity0))
            f.write('\n')
            f.write('reacall is ')
            f.write(str(recall0))
            f.write('\n')
            f.write('fscore is ')
            f.write(str(fscore0))
            f.write('\n')
            f.write('MACC is ')
            f.write(str(MACC0))


        with open(save_path1, 'a+') as f:
            f.write('----------------------------------------------------------------')
            f.write('\n')
            f.write('num mfcc is : ')
            f.write(data_subset[i])
            f.write('\n')
            f.write('sensitivity is ')
            f.write(str(sensitivity1))
            f.write('\n')
            f.write('reacall is ')
            f.write(str(recall1))
            f.write('\n')
            f.write('fscore is ')
            f.write(str(fscore1))
            f.write('\n')
            f.write('MACC is ')
            f.write(str(MACC1))

        with open(save_path2, 'a+') as f:
            f.write('----------------------------------------------------------------')
            f.write('\n')
            f.write('num mfcc is : ')
            f.write(data_subset[i])
            f.write('\n')
            f.write('sensitivity is ')
            f.write(str(sensitivity2))
            f.write('\n')
            f.write('reacall is ')
            f.write(str(recall2))
            f.write('\n')
            f.write('fscore is ')
            f.write(str(fscore2))
            f.write('\n')
            f.write('MACC is ')
            f.write(str(MACC2))




'''
    feature_name = [
                        'mfcc_systole_1','mfcc_systole_2','mfcc_systole_3','mfcc_systole_4','mfcc_systole_5','mfcc_systole_6',\
                        'mfcc_systole_7','mfcc_systole_8','mfcc_systole_9','mfcc_systole_10','mfcc_systole_11','mfcc_systole_12', \
                        'mfcc_systole_13', 'mfcc_systole_14', 'mfcc_systole_15', 'mfcc_systole_16', 'mfcc_systole_17', 'mfcc_systole_18', \
                        'mfcc_systole_19', 'mfcc_systole_20', 'mfcc_systole_21', 'mfcc_systole_22', 'mfcc_systole_23','mfcc_systole_24', \
                        'mfcc_systole_25', 'mfcc_systole_26', 'mfcc_systole_27', 'mfcc_systole_28', 'mfcc_systole_29', 'mfcc_systole_30', \
                        'mfcc_systole_31', 'mfcc_systole_32', 'mfcc_systole_33', 'mfcc_systole_34', 'mfcc_systole_35', 'mfcc_systole_36', \
                        #'mfcc_systole_37', 'mfcc_systole_38', 'mfcc_systole_39', 'mfcc_systole_40', 'mfcc_systole_41', 'mfcc_systole_42', \
                        #'mfcc_systole_43', 'mfcc_systole_44', 'mfcc_systole_45', 'mfcc_systole_46', 'mfcc_systole_47', 'mfcc_systole_48', \
                        'mfcc_diastole_1', 'mfcc_diastole_2', 'mfcc_diastole_3', 'mfcc_diastole_4', 'mfcc_diastole_5', 'mfcc_diastole_6', \
                        'mfcc_diastole_7', 'mfcc_diastole_8', 'mfcc_diastole_9', 'mfcc_diastole_10', 'mfcc_diastole_11', 'mfcc_diastole_12', \
                        'mfcc_diastole_13', 'mfcc_diastole_14', 'mfcc_diastole_15', 'mfcc_diastole_16', 'mfcc_diastole_17', 'mfcc_diastole_18', \
                        'mfcc_diastole_19', 'mfcc_diastole_20', 'mfcc_diastole_21', 'mfcc_diastole_22', 'mfcc_diastole_23', 'mfcc_diastole_24', \
                        'mfcc_diastole_25', 'mfcc_diastole_26', 'mfcc_diastole_27', 'mfcc_diastole_28', 'mfcc_diastole_29', 'mfcc_diastole_30', \
                        'mfcc_diastole_31', 'mfcc_diastole_32', 'mfcc_diastole_33', 'mfcc_diastole_34', 'mfcc_diastole_35', 'mfcc_diastole_36', \
                        #'mfcc_diastole_37', 'mfcc_diastole_38', 'mfcc_diastole_39', 'mfcc_diastole_40', 'mfcc_diastole_41', 'mfcc_diastole_42', \
                        #'mfcc_diastole_43', 'mfcc_diastole_44', 'mfcc_diastole_45', 'mfcc_diastole_46', 'mfcc_diastole_47', 'mfcc_diastole_48', \
                        'feature_label']
'''