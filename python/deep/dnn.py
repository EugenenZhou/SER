import keras
import numpy as np
import pandas as pd
import datetime
import time
from keras.models import Model,Sequential
from keras.layers import Conv1D,Flatten,Dense,BatchNormalization,Dropout,Activation
from sklearn.utils import shuffle
from keras import optimizers
from keras import regularizers
from keras import callbacks
from keras.callbacks import EarlyStopping
import random


batchsize=256
epochs=120
cat_num=4
# np.random.seed(4)

def creatmodel(train_x, train_y, vaild_x, vaild_y,dim_length,save_path):
    sql=Sequential()
    rm = keras.initializers.RandomNormal(mean=0.0, stddev=1)

    sql.add(Dense(1024,input_shape=(dim_length,),kernel_initializer=rm))
    sql.add(BatchNormalization(axis=1))
    sql.add(Activation('relu'))
    # sql.add(Dropout(rate=0.2))
    #
    sql.add(Dense(1024, kernel_initializer=rm, bias_initializer=rm))
    sql.add(BatchNormalization(axis=1))
    sql.add(Activation('relu'))
    # sql.add(Dropout(rate=0.5))
    #
    # sql.add(Dense(1024, kernel_initializer=rm, bias_initializer='zeros'))
    # sql.add(BatchNormalization(axis=1))
    # sql.add(Activation('relu'))
    # sql.add(Dropout(rate=0.5))

    # sql.add(Dense(1024, kernel_initializer=rm, bias_initializer='zeros'))
    # sql.add(BatchNormalization(axis=1))
    # sql.add(Activation('relu'))
    # # sql.add(Dropout(rate=0.5))

    sql.add(Dense(cat_num, kernel_initializer='random_uniform', bias_initializer='zeros'))
    sql.add(Activation('softmax'))

    rms = keras.optimizers.RMSprop(lr=0.01)

    sql.compile(loss='categorical_crossentropy',
                optimizer=rms,
                metrics=['categorical_accuracy', ])

    ESt = keras.callbacks.EarlyStopping(monitor='val_loss', patience=11)
    RLP = keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-14)
    MCp = keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_categorical_accuracy',
                                          save_best_only=True,mode='max')
    print(sql.summary())
    print('Train...')

    sql.fit(x=train_x, y=train_y, validation_data=(vaild_x, vaild_y), batch_size=batchsize, epochs=epochs, shuffle=True,
            callbacks=[RLP, ESt, MCp])
    return sql

def load_data(data_name):
    if data_name=='op384':
        traindata = np.array(pd.read_csv('F:/data/opsm384/train.csv'))
        vailddata=np.array(pd.read_csv('F:/data/opsm384/vaild.csv'))
        testdata = np.array(pd.read_csv('F:/data/opsm384/test.csv'))
        return traindata,vailddata,testdata
    if data_name=='op1580':
        traindata = np.array(pd.read_csv('F:/data/opsm1580/zcl/ztrain.csv'))
        vailddata=np.array(pd.read_csv('F:/data/opsm1580/zcl/zvaild.csv'))
        testdata = np.array(pd.read_csv('F:/data/opsm1580/zcl/ztest.csv'))
        return traindata,vailddata,testdata
    if data_name=='sxop1580':
        traindata = np.array(pd.read_csv('F:/data/opsm1580/sx300train1580.csv'))
        vailddata=np.array(pd.read_csv('F:/data/opsm1580/sx300vaild1580.csv'))
        testdata = np.array(pd.read_csv('F:/data/opsm1580/sx300test1580.csv'))
        return traindata,vailddata,testdata


#数据预处理
def preprocessing(traindata, vailddata, testdata):
    ##训练数据
    x_train = traindata[:, 0:-1]
    y_train = traindata[:, -1]
    x_train = np.reshape(x_train, [-1, dim_length])
    y_train = keras.utils.to_categorical(y_train, cat_num)

    ###测试数据
    x_test = testdata[:, 0:-1]
    y_test = testdata[:, -1]
    x_test = np.reshape(x_test, [-1,dim_length])
    y_test = keras.utils.to_categorical(y_test, cat_num)

    ###验证数据
    x_vaild = vailddata[:, 0:-1]
    y_vaild = vailddata[:, -1]
    x_vaild = np.reshape(x_vaild, [-1,dim_length])
    y_vaild = keras.utils.to_categorical(y_vaild, cat_num)
    return x_train,y_train,x_vaild,y_vaild,x_test,y_test

def predict(x):
    cat=0
    p_cat=0
    b_value=0
    for values in x:
        for value in values:
            if value >=b_value:
                b_value=value
                p_cat=cat
            cat+=1
    return p_cat

def predictx(x):
    cat=0
    p_cat=0
    b_value=0
    for values in x:
        for value in values:
            if value >=b_value:
                b_value=value
                p_cat=cat
            cat+=1
    return p_cat,b_value



if __name__ == '__main__':
    # total_acc=[]
    dim_length = 300
    ks=[]
    savepath1='F:/savep/m1.hdf5'


    traindata, vailddata, testdata = load_data('sxop1580')
    # traindata1=traindata[:,0:dim_length]
    # traindata2=traindata[:,300:301]
    # traindata=np.concatenate((traindata1,traindata2),axis=1)
    # vailddata1=vailddata[:,0:dim_length]
    # vailddata2=vailddata[:,300:301]
    # vailddata=np.concatenate((vailddata1,vailddata2),axis=1)
    # testdata1=testdata[:,0:dim_length]
    # testdata2=testdata[:,300:301]
    # testdata=np.concatenate((testdata1,testdata2),axis=1)
    x_train, y_train, x_vaild, y_vaild,x_test, y_test = preprocessing(traindata, vailddata,testdata)

    model = creatmodel(train_x=x_train, train_y=y_train, vaild_x=x_vaild, vaild_y=y_vaild, dim_length=dim_length,
                       save_path=savepath1)
    acc,p=model.evaluate(x=x_test,y=y_test)
    print(acc,p)













