#提取语音的mfcc或者fb特征
#默认提取fb，设置fb_ext不等于1时提取mfcc

import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt


def fbmfcc(path,fb_ext=1):
    #读取语音信号
    sample_rate,signal=scipy.io.wavfile.read(path) #sample_rate为采样率 signal为信号段
    # signal=signal[0:int(14.5*sample_rate)]#选取前14.5秒的信号
    signal = signal[0:int(3.5 * sample_rate)]  # 选取前3.5秒的信号

    x=np.arange(0,3.5,1/sample_rate)
    #plt.figure()
    #plt.plot(x,signal)
    #plt.show()

    #预加重
    #原因：由于在传输过程中，信号的损失会很大，为了使接受端获得比较好的波形，需对受损的信号进行补偿
    #目的：增加信号的高频成分，因为传输通道的低通滤波特性，使得低频成分衰减少，高频成分衰减大
    #对在发送端发送信号的高频分量进行补偿的方法，增大信号跳变边沿后第一个bit的幅度（上升沿或下降沿）
    #优势：1.平衡频谱，使高低频分量相同（一般是高频比低频少）2.避免在傅里叶变换时出现问题 3.改善信噪比
    #y(t)=x(t)-ax(t-1) a通常取0.97

    pre_emphasis=0.97 #系数
    emphasis_signal=np.append(signal[0],signal[1:]-pre_emphasis*signal[:-1])#预加重信号，第一个不变，第二个信号为前一个信号减去当前信号
    #plt.plot(x,emphasis_signal)
    #plt.show()

    #分帧
    #原因：频率会随时间的流逝不断改变，即对整个语音信号进行傅里叶变化是没有意义的，因为可能会损失频率轮廓，信号不平稳
    #假定短时的信号频率是固定的，平稳的
    #对预加重后的信号段分成一个一个短时的帧，在对每个帧进行处理，从而得到一个对于信号频率更好的近似
    #常用帧长为20-40ms配上50%（+-10%）的覆盖率，普遍取帧长25ms，帧移10ms

    frame_size=0.025#帧长
    frame_stride=0.01#帧移
    frame_length,frame_step=sample_rate*frame_size,sample_rate*frame_stride#变换信号为时间
    signal_length=len(emphasis_signal)#信号长度
    frame_length=int(round(frame_length))#取整
    frame_step=int(round(frame_step))#取整
    num_frames=int(np.ceil(float(np.abs(signal_length-frame_length))/frame_step))#保证至少有一帧

    pad_signal_length=num_frames*frame_step+frame_length#补足后的信号长度
    zero_padding=np.zeros((pad_signal_length-signal_length))#补零
    pad_signal=np.append(emphasis_signal,zero_padding)#补零后的新的信号

    indices=np.tile(np.arange(0,frame_length),(num_frames,1))+np.tile(np.arange(0,num_frames*frame_step,frame_step),(frame_length,1)).T#排序出矩阵，很精髓的代码！
    frames=pad_signal[indices.astype(np.int32,copy=False)]#分帧完成

    #加窗
    #fft算法会认为采样来的信号是无限长的周期信号，即信号会不断重复，从头到尾，从尾到头，这样可能会导致波形不连续的情况发生，从而频谱泄露的情况发生
    #分帧之后需要加窗，配合窗函数，使得原本不太平滑的地方看上去平滑了，一般采用hamming窗
    #hamming窗：w(n)=0.54-0.46*cos((2*pi*n)/(N-1)) 其中0<=n<=N-1,N是窗口长度，

    hamming=np.tile((0.54-0.46*np.cos((2*np.pi*np.arange(0,frame_length))/(frame_length-1))),(num_frames,1))#乘上窗口函数
    frames=frames*hamming
    #frames=frames*np.hamming(frame_length)#或者写成这样

    #傅里叶变换和能量谱
    #对加完窗函数的波进行傅里叶变换，来计算频谱，短时傅里叶变换
    #N点的FFT 一般N取256和512
    #P=(|FFT(xi)|^2)/(N) xi是信号x的第i的窗

    NFFT=512
    mag_frames=np.absolute(np.fft.rfft(frames,NFFT))#FFT变换并计算绝对值
    pow_frames=((1.0/NFFT)*((mag_frames)**2))#计算频谱


    #FB特征提取
    #应用三角形滤波器组对求得频谱进行滤波，Mel-scale从频谱中提出频带。通常选用40个滤波器
    #Mel-scale旨在从模拟非线性人耳对声音的感知，对低频更有判别力，对高频更少
    #频率与Mel的相互转化 m=2595log10(1+f/700)或f=700(10^(m/2595)-1)
    #每个滤波器都是三角形的，0-1，1为中心频率，线性下降到两个相邻滤波器时为0
    #
    #      |          0                 k<f(m-1)
    #      | (k-f(m-1))/(f(m)-f(m-1))   f(m-1)<=k<f(m)
    #Hm(k)=|          1                 k=f(m)
    #      | (f(m+1)-k)/(f(m+1)-f(m))   f(m)<k<=f(m+1)
    #      |          0                 k>f(m-1)
    #
    #




    nfilt=40#fb三角滤波器的数量
    low_freq_mel=0#最低频率
    high_freq_mel=(2595*np.log10(1+(sample_rate/2)/700))#将HZ转化为Mel
    mel_points=np.linspace(low_freq_mel,high_freq_mel,nfilt+2)#返回均匀分布的数字，即均匀分布mel的值
    hz_points=(700*(10**(mel_points/2595)-1))#将Mel转化为HZ
    bin=np.floor((NFFT+1)*hz_points/sample_rate)

    fbank=np.zeros((nfilt,int(np.floor(NFFT/2+1))))
    for m in range(1,nfilt+1):
        f_m_minus=int(bin[m-1])#左
        f_m=int(bin[m])#中
        f_min_plus=int(bin[m+1])#右

        for k in range(f_m_minus,f_m):
            fbank[m-1,k]=(k-bin[m-1])/(bin[m]-bin[m-1])
        for k in range(f_m,f_min_plus):
            fbank[m-1,k]=(bin[m+1]-k)/(bin[m+1]-bin[m])
    filter_banks=np.dot(pow_frames,fbank.T)#矩阵相乘
    filter_banks=np.where(filter_banks==0,np.finfo(float).eps,filter_banks)#数值稳定性
    filter_banks=20*np.log10(filter_banks)#db



    #MFCC
    #fb的参数计算是高度相关的，导致在某些机器学习方法中会出现问题
    #对fb特征进行DCT的变换，来降低这些相关性，产生一个滤波器组的压缩表示
    #倒谱系数一般取2-13左右
    #
    #
    num_ceps=13#mfcc倒谱系数
    mfcc=dct(filter_banks,type=2,axis=1,norm='ortho')[:,1:(num_ceps+1)]#
    (nframes,ncoeff)=mfcc.shape
    n=np.arange(ncoeff)
    lift=1+(num_ceps/2)*np.sin(np.pi*n/num_ceps)
    mfcc+=lift

    #均值归一化
    #平衡频谱改善信噪比，减去每个系数的均值
    filter_banks=filter_banks-(np.mean(filter_banks,axis=0)+1e-8)

    mfcc=mfcc-(np.mean(mfcc,axis=0)+1e-8)

    if fb_ext==1:
        return filter_banks
    else:
        return mfcc

































