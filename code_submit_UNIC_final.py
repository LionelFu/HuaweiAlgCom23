import os, time
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift, ifft, ifftshift
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
import statsmodels.api as sm
import pywt
from scipy.ndimage.filters import median_filter

def EstBreathRate(Cfg, CSI):
    '''
    估计每个4D CSI样本的呼吸率，需参设者自行设计
    :param Cfg: CfgX文件中配置信息，dict
    :param CSI: 4D CSi数据
    :iSamp: 本次估计Sample集合中第iSamp个样本
    :return:呼吸率估计结果， 长度为Np的numpy数组
    '''
    #########以下代码，参赛者用自己代码替代################
    res_list = []
    for sample_idx in tqdm(range(Cfg['Nsamp'])):
        csi_Nt = Cfg['Nt'][sample_idx]
        csi_T = Cfg['Tdur'][sample_idx]
        signal_res_sc = []
        peak_diff_list = []
        antes = np.arange(Cfg['Nrx'])
        pairs = []
        data_list = []
        sens_list = []
        quantification_list = []
        for ante1 in antes:
            for ante2 in antes[ante1+1:]:
                pairs.append((ante1, ante2))
        t = 0
        for (base_rx, target_rx) in pairs:
            angle_diff_list, ap_list = angleDiff(CSI, Cfg, sample_idx, target_rx, base_rx)

            # angle_diff_list = ap_list
            for src in range(Cfg['Nsc']):
                filtered_data = filter(Cfg, sample_idx, angle_diff_list[src, :])       # 滤波后相位差
                # filtered_data = filter(Cfg, sample_idx, ap_list[src, :])
                fft_signal = np.abs(fftshift(fft(filtered_data, axis=-1)))    # 相位差频谱
                
                _, counts = np.unique(filtered_data, return_counts=True)
                quantification_list.append(np.sort(counts)[-3:].sum())  # 量化程度
                
                fft_signal_cut = fft_signal[len(fft_signal)//2:]
                # spectrum_list.append(fft_signal_cut)

                data_list.append(filtered_data)

                sens_list.append(np.abs(filtered_data-np.mean(filtered_data)).mean())
                
                # 检测频谱所有峰值
                diff = np.diff(fft_signal_cut)  # 计算差分
                peaks = np.where((diff[:-1] > 0) & (diff[1:] < 0))[0]+1  # 峰值点

                # 去除有效范围外峰值
                peaks = peaks[peaks>min_fre*csi_T]
                peaks = peaks[peaks<max_fre*csi_T]
                # if peaks[0]<0.1*csi_T:    # 如果存在极低频强分量
                t += 1
                if len(peaks) == 1:     # 只有一个峰
                    max_peak = peaks[0]
                    signal_result = ((((max_peak))/csi_T))*60   # 频谱最大值对应bpm
                    signal_res_sc.append(signal_result)
                    peak_diff_list.append(0)
                    continue
                elif len(peaks) == 0:   # 没有峰
                    signal_res_sc.append(0)
                    peak_diff_list.append(0) 
                    continue
                peak_values = fft_signal_cut[peaks]     # 所有峰的值
                sorted_idxs = np.argsort(peak_values)
                max_idx = sorted_idxs[-1]   # 最大峰对应的id
                top_k_idx = sorted_idxs[:-1]

                peak_diff = np.abs(peak_values[max_idx]-np.mean(peak_values[top_k_idx]))/peak_values[max_idx]
            
                peak_diff_list.append(peak_diff)
                max_peak = peaks[max_idx]
                signal_result = (max_peak/csi_T)*60   # 频谱最大值对应频率
                signal_res_sc.append(signal_result)
        signal_res_sc = np.array(signal_res_sc)
        peak_diff_list = np.array(peak_diff_list)
        peak_diff_list[np.array(quantification_list)>=50] = 0   # 去除量化结果
        data_list = np.array(data_list)
        # spectrum_list = np.vstack(spectrum_list)

        # if Cfg['Np'][sample_idx]==1:
        # 加权投票
        resu, counts = np.unique(np.array(signal_res_sc), return_counts=True)   # 不同结果以及数量

        weight_1 = counts/counts.sum()
        weight_2 = np.array([peak_diff_list[signal_res_sc==r].sum() for r in resu])
        weight = weight_1*weight_2
        weight = weight*np.exp(-(((resu - 20) **2) / (2 * 5 ** 2)))
        Np = Cfg['Np'][sample_idx]
        signal_res = np.sort(resu[np.argsort(weight)[-Np:]])
        res_list.append(signal_res)


    #########样例代码中直接返回随机数作为估计结果##########
    # return np.random.rand(Cfg['Np'][iSamp]) * 100
    return res_list


def RMSEerr(EstIn, GtIn):
    '''
    计算RMSE误差
    :param Est: 估计的呼吸率，1D
    :param Gt: 测量的呼吸率，1D
    :return: rmse误差
    '''
    Est = np.concatenate(EstIn)
    Gt = np.concatenate(GtIn)
    if np.size(Est) != np.size(Gt):
        print("呼吸率估计数目有误，输出无效!")
        return -1
    rmse = np.sqrt(np.mean(np.square(Gt - Est)))
    return rmse

def CsiFormatConvrt(Hin, Nrx, Ntx, Nsc, Nt):
    '''
    csi格式转换，从2D [NT x (Nsc*NRx*NTx)]转为4D [NRx][NTx][NSc][NT]
    '''
    Hout = np.reshape(Hin, [Nt, Nsc, Nrx, Ntx])
    Hout = np.transpose(Hout, [2, 3, 1, 0])
    return Hout

def EstRRByWave(wa, fs):
    Rof = 3
    n = 2**(np.ceil(np.log2(len(wa)))+Rof)
    blow, bhigh = [8,50] #参赛者自行决策呼吸估计区间
    low = int(np.ceil(blow/60/fs*n))
    high = int(np.floor(bhigh/60/fs*n))
    spec = abs(np.fft.fft(wa-np.mean(wa), int(n)))
    stat = 2*2**Rof
    tap = np.argwhere(spec[stat:int(n/2)] == max(spec[low: high]))[0] + stat
    return tap/n*fs*60


def hampel(data, window_size, num_sigma=3):
    median = median_filter(data, size=(1, window_size))
    deviation = np.median(np.abs(data - median), axis=-1)
    threshold = num_sigma * deviation
    mask = np.abs(data - median) > threshold[..., np.newaxis]
    filtered_data = np.where(mask, median, data)
    return filtered_data



def norm_signal(CSI_s, sample_idx, antenna_idx):
    plot_phase_1 = CSI_s[sample_idx][antenna_idx, :, :, :] / np.abs(CSI_s[sample_idx][antenna_idx, :, :, :])
    # plot_phase_1 = plot_phase_1 / np.repeat(np.average(plot_phase_1, axis=-1)[:, :, np.newaxis], repeats=self.Cfg['Nt'][sample_idx], axis=-1)
    return plot_phase_1.squeeze()

def angleDiff(CSI_s, Cfg, sample_idx, nx_target, nx_base):
    angle_diff_list = []

    # 第一根天线和第三根天线，第一个子载波
    base_signal = CSI_s[sample_idx][nx_base, :, :, :].squeeze()
    target_signal = CSI_s[sample_idx][nx_target, :, :, :].squeeze()
    base_ap = np.abs(base_signal)
    target_ap = np.abs(target_signal)
    base_ap[base_ap==0] = np.mean(base_ap)
    target_ap[target_ap==0] = np.mean(target_ap)
    base_signal = base_signal/base_ap
    target_signal = target_signal/target_ap

    base_signal = base_signal/(np.repeat(np.average(base_signal, axis=-1)[:, np.newaxis], Cfg['Nt'][sample_idx], axis=-1))
    target_signal = target_signal/np.repeat(np.average(target_signal, axis=-1)[:, np.newaxis], Cfg['Nt'][sample_idx], axis=-1)
    target_angle = np.angle(target_signal, deg=True)
    base_angle = np.angle(base_signal, deg=True)
    angle_diff_list = target_angle-base_angle
    # angle_diff_list[np.where(angle_diff_list<-180)] += 360
    # angle_diff_list[np.where(angle_diff_list>180)] -= 360
    angle_diff_list = np.min(np.concatenate([angle_diff_list[:, :, np.newaxis], angle_diff_list[:, :, np.newaxis]+360, angle_diff_list[:, :, np.newaxis]-360], axis=-1), axis=-1)
    return angle_diff_list, base_ap

def adjust_window_means(signal, window_size, threshold):
    # 计算信号的均值
    signal_mean = np.mean(signal)
    adjusted_signal = np.copy(signal)
    signal_mad = np.abs(signal-signal.mean()).mean()
    n = len(signal)

    # 滑动窗口处理
    for i in range(0, n, window_size):
        window = signal[i:i+window_size]
        # window_mad = np.abs(window-window.mean()).mean()
        window_mad = np.abs(window.mean() - signal_mean)
        # print(window_mad)
        if window_mad/signal_mad > threshold:
            adjusted_signal[i:i+window_size] = (2*((window-window.min())/(window.max()-window.min()))-1)*signal_mad+signal_mean
            # print(adjusted_signal[i:i+window_size].min())
            # # 将窗口的均值调整为信号的均值
            # adjusted_window_mean = (window.mean() - signal_mean)
            # adjusted_signal[i:i+window_size] -= adjusted_window_mean

    return adjusted_signal


def filter(Cfg, sample_idx, data):
    sample_fre = (Cfg['Nt'][sample_idx]-1)/Cfg['Tdur'][sample_idx]
    # 滤波器设计
    base_diff = data.copy()

    base_diff = hampel(base_diff[np.newaxis, :], 21).squeeze()   # 去除高频分量

    base_diff = adjust_window_means(base_diff, int(sample_fre*2.5), threshold=1)

    
    # base_diff = adjust_window_means(base_diff, 100, threshold=1)
    
    coeffs = pywt.wavedec(base_diff, 'db4', level=3)
    base_diff = pywt.waverec([coeffs[0], None,  None, None],  'db4')

    filteredData = base_diff
    
    filteredData = filteredData-np.mean(filteredData)     # 去除零频点

    return filteredData

class SampleSet:
    "样本集基类"
    Nsamples = 0 #总样本数类变量

    def __init__(self, name, Cfg, CSIs):
        self.name  = name
        self.Cfg   = Cfg
        self.CSI   = CSIs #所有CSI
        self.CSI_s = []   #sample级CSI
        self.Rst   = []
        self.Wave  = []   # 测量所得呼吸波形，仅用于测试
        self.Gt    = []   # 测量呼吸率，仅用于测试
        self.GtRR  = []   # 测量波形呼吸率，仅用于测试
        SampleSet.Nsamples += self.Cfg['Nsamp']

    def estBreathRate(self):
        # BR = []
        # CSI数据整形，建议参赛者根据算法方案和编程习惯自行设计，这里按照比赛说明书将CSI整理成4D数组，4个维度含义依次为收天线，发天线，子载波，时间域测量索引
        Nt = [0] + list(accumulate(self.Cfg['Nt']))
        for ii in range(self.Cfg['Nsamp']):
            self.CSI_s.append(CsiFormatConvrt(self.CSI[Nt[ii]:Nt[ii+1],:], self.Cfg['Nrx'],
                                              self.Cfg['Ntx'], self.Cfg['Nsc'], self.Cfg['Nt'][ii]))
        # for ii in range(self.Cfg['Nsamp']):
        #     br = EstBreathRate(self.Cfg, self.CSI_s[ii], ii)  ## 呼吸率估计
        #     BR.append(br)
        BR = EstBreathRate(self.Cfg, self.CSI_s)
        self.Rst = BR

    def getRst(self):
        return self.Rst

    def getEstErr(self):
        rmseE = RMSEerr(self.Rst, self.Gt)
        print("<<<RMSE Error of SampleSet file #{} is {}>>>\n".format(self.name, rmseE))
        return rmseE

    def setGt(self, Gt):
        self.Gt = Gt

    def setWave(self, wave):
        #此处按照样例排布Wave波形，如self.Wave[iSamp][iPerson]['Wave'] 第iSamp个样例的第iPerson个人的波形
        NP = [0] + list(accumulate(self.Cfg['Np']))
        for ii in range(self.Cfg['Nsamp']):
            self.Wave.append(wave[NP[ii]:NP[ii+1]])

    def estRRByWave(self):
        for ii in range(len(self.Wave)):
            RR = []
            for jj in range(len(self.Wave[ii])):
                wa = abs(self.Wave[ii][jj]['Wave'])
                para = self.Wave[ii][jj]['Param']
                fs = (para[0]-1) / para[1]
                RR.append(EstRRByWave(wa, fs))
            #print("rr = ", RR, ", ii ", ii, ", jj", jj)
            self.GtRR.append(np.array(RR))
        return self.GtRR

def FindFiles(PathRaw):
    dirs = os.listdir(PathRaw)
    names = []  #文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!= -1 and f.endswith('.txt'):
            print('Now reading file of {} ...\n'.format(f))
            names.append(f.split('CfgData')[-1].split('.txt')[0])
    return names, files

def CfgFormat(fn):
    a = []
    with open(fn, 'r') as f:
        for line in f:
            d = np.fromstring(line, dtype = float, sep = ' ')#[0]
            a.append(d)
    return {'Nsamp':int(a[0][0]), 'Np': np.array(a[1],'int'), 'Ntx':int(a[2][0]), 'Nrx':int(a[3][0]),
            'Nsc':int(a[4][0]), 'Nt':np.array(a[5],'int'), 'Tdur':a[6], 'fstart':a[7][0], 'fend':a[8][0]}

def ReadWave(fn):
    Wave = []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            wa = {}
            wa['Param'] = np.fromstring(lines[i].strip(), dtype=float, sep = ' ')
            wa['Wave'] = np.fromstring(lines[i+1].strip(), dtype=int, sep = ' ')
            Wave.append(wa)
    return Wave


if __name__ == '__main__':
    print("<<< Welcom to 2023 Huawei Algorithm Contest! This is demo code. >>>\n")
    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
    PathSet = {0:"./TestData2", 1:"./CompetitionData1", 2:"./CompetitionData2", 3:"./CompetitionData3", 4:"./CompetitionData4"}
    PrefixSet = {0:"Test" , 1:"Round1", 2:"Round2", 3:"Round3", 4:"Round4"}

    ## params Test
    Ridx = 3 # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

    tStart = time.perf_counter()
    ## 1查找文件
    names= FindFiles(PathRaw) # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中

    dirs = os.listdir(PathRaw)
    names = []  # 文件编号
    files = []
    for f in sorted(dirs):
        if f.endswith('.txt'):
            files.append(f)
    for f in sorted(files):
        if f.find('CfgData')!=-1 and f.endswith('.txt'):
            names.append(f.split('CfgData')[-1].split('.txt')[0])

    ## 2创建对象并处理
    Rst = []
    Gt  = []
    for na in names: #[names[0]]:#
        # 读取配置及CSI数据
        Cfg = CfgFormat(PathRaw + '/' + Prefix + 'CfgData' + na + '.txt')
        csi = np.genfromtxt(PathRaw + '/' + Prefix + 'InputData' + na + '.txt', dtype = float)
        CSI = csi[:,0::2] + 1j* csi[:,1::2]

        samp = SampleSet(na, Cfg, CSI)
        del CSI

        max_fre = 0.8
        min_fre = 0.1

        if Ridx == 0: ## 对于测试数据，参赛选手可以读取真实呼吸波形用于分析
            Wave = ReadWave(PathRaw + '/' + Prefix + 'BreathWave' + na + '.txt')
            samp.setWave(Wave)
            samp.estRRByWave() #计算呼吸率，依赖波形数据

        # 对于测试数据，参赛选手可基于真实呼吸数据计算估计RMSE
        if Ridx == 0:
            with open(PathRaw + '/' + Prefix + 'GroundTruthData' + na + '.txt', 'r') as f:
                    gt = [np.fromstring(arr.strip(), dtype=float, sep = ' ') for arr in f.readlines()]
            samp.setGt(gt)
            # samp.getEstErr()  ## 计算每个输入文件的RMSE
            Gt.extend(gt)
        
        # 计算并输出呼吸率
        samp.estBreathRate()  ## 请进入该函数以找到编写呼吸率估计算法的位置
        rst = samp.getRst()
        Rst.extend(rst)

        # 3输出结果：各位参赛者注意输出值的精度
        with open(PathRaw + '/' + Prefix + 'OutputData' + na + '.txt', 'w') as f:
            [np.savetxt(f, np.array(ele).reshape(1, -1), fmt = '%.6f', newline = '\n') for ele in rst]

        
        

    if Ridx == 0: # 对于测试数据，计算所有样本的RMSE
        rmseAll = RMSEerr(Rst, Gt)
        print("<<<RMSE Error of all Samples is {}>>>\n".format(rmseAll))

    ## 4统计时间
    tEnd = time.perf_counter()
    print("Total time consuming = {}s".format(round(tEnd-tStart, 3)))

