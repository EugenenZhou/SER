import os
import random
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import keras
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import Convolution2D,AveragePooling2D,Flatten,BatchNormalization,MaxPooling2D
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras.backend as K

epochs = 120#轮回次数
cat_num = 4#分类数
batchsize = 256
labelzero = 0#起始类标
total_acc=[]
acc1=0
std=0

def creatmodel(train_x,train_y,vaild_x,vaild_y,save_path,std):
    sql=Sequential()

    rm = keras.initializers.RandomNormal(mean=0.0,stddev=0.24)

    #strides = (1, 1)
    sql.add(Convolution2D(1,kernel_size=(1,1),strides=(1,1),input_shape=(squlength,datadim,1),
                          bias_regularizer=regularizers.l2(0.001),padding='same',kernel_initializer=rm))
    #sql.add(Dropout(rate=0.5))
    # sql.add(BatchNormalization())
    # sql.add(Activation('relu'))
    #sql.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # sql.add(Convolution2D(32,kernel_size=(2,2),strides=(2,2),bias_regularizer=regularizers.l2(0.001),
    #                       padding='same',kernel_initializer=rm))
    # sql.add(BatchNormalization())
    # sql.add(Activation('relu'))
    #sql.add(Dropout(rate=0.5))


    #sql.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    sql.add(Flatten())

    sql.add(Dense(1024,kernel_initializer=rm,bias_initializer='zeros'))
    sql.add(BatchNormalization(axis=1))
    sql.add(Activation('relu'))
    sql.add(Dropout(rate=0.5))

    sql.add(Dense(1024,kernel_initializer=rm,bias_initializer='zeros'))
    sql.add(BatchNormalization(axis=1))
    sql.add(Activation('relu'))
    sql.add(Dropout(rate=0.5))

    sql.add(Dense(cat_num,kernel_initializer='random_uniform',bias_initializer='zeros'))
    sql.add(Activation('softmax'))

    # hitory = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='auto')
    rms = keras.optimizers.RMSprop(lr=0.01,decay=0.99)

    sql.compile(loss='categorical_crossentropy',
                  optimizer=rms,
                  metrics=['categorical_accuracy',])

    ESt=keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
    RLP=keras.callbacks.ReduceLROnPlateau(patience=4,min_lr=0.00000001)

    print(sql.summary())
    print('Train...')

    sql.fit(x=train_x,y=train_y,validation_data=(vaild_x,vaild_y),batch_size=batchsize,epochs=epochs,shuffle=True,callbacks=[RLP,ESt])
    return sql


def load_data(data_name):
    if data_name=='12mfcc':
        traindata = np.array(pd.read_csv('F:/data/12mfcc/12train.csv'))
        vailddata = np.array(pd.read_csv('F:/data/12mfcc/12vaild.csv'))
        testdata = np.array(pd.read_csv('F:/data/12mfcc/12test.csv'))
        return traindata,vailddata,testdata


def preprocess(traindata,vailddata,testdata):
    ###traindata
    x_train = traindata[:, 0:-1]
    y_train= traindata[:, -1]
    x_traindata_one = np.reshape(x_train, [-1, squlength, datadim,1])
    y_traindata_one = y_train[::squlength]

    x_train, x_trainlabel = shuffle(x_traindata_one, y_traindata_one)
    x_train = np.array(x_train)
    x_trainlabel = np.array(x_trainlabel)-labelzero
    x_trainlabel = keras.utils.to_categorical(x_trainlabel, cat_num)


    ###testdata
    x_test=testdata[:,0:-1]
    y_test=testdata[:,-1]
    x_testdata_one = np.reshape(x_test, [-1, squlength, datadim,1])
    y_testdata_one = y_test[::squlength]

    x_test, x_testlabel = shuffle(x_testdata_one, y_testdata_one)
    x_test = np.array(x_test)
    x_testlabel = np.array(x_testlabel)-labelzero
    x_testlabel = keras.utils.to_categorical(x_testlabel, cat_num)


    ###vailddata
    x_vaild=vailddata[:,0:-1]
    y_vaild=vailddata[:,-1]
    x_vailddata_one = np.reshape(x_vaild, [-1, squlength, datadim,1])
    y_vailddata_one = y_vaild[::squlength]

    x_vaild, x_vaildlabel = shuffle(x_vailddata_one, y_vailddata_one)
    x_vaild = np.array(x_vaild)
    x_vaildlabel = np.array(x_vaildlabel)-labelzero
    x_vaildlabel = keras.utils.to_categorical(x_vaildlabel, cat_num)

    return x_train,x_trainlabel,x_vaild,x_vaildlabel,x_test,x_testlabel

if __name__ == '__main__':
    traindata, vailddata, testdata=load_data('12mfcc')
    path = 'D:/savep/m1.hd5f'
    squlength = 200  # 样本长度
    datadim = 12  # 样本维度数
    x_train, x_trainlabel, x_vaild, x_vaildlabel, x_test, x_testlabel=preprocess(traindata, vailddata, testdata)
    model = creatmodel(train_x=x_train, train_y=x_trainlabel,vaild_x=x_vaild,vaild_y=x_vaildlabel,save_path=path,std=std)
    score,acc = model.evaluate(x_test,x_testlabel)
    print(acc)










