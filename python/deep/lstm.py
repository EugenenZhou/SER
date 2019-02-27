import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import keras
from keras.models import Sequential
from keras.layers import LSTM,Masking,TimeDistributed
from keras.layers.core import Dense,Activation
from keras.callbacks import EarlyStopping


epochs=40
avgacc=0
numclass=4
batch_size=400
labelzero=0#类标为0就为0
totalnum=10
scoreall=[]
accall=[]



def creatmodel(x_train,y_train,x_vail,y_vail):
    sql = Sequential()
    sql.add(LSTM(512, recurrent_dropout=0.5, input_shape=(squlength, datadim),
                   kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.09, seed=None),
                   return_sequences=True))
    sql.add(LSTM(256, recurrent_dropout=0.5, return_sequences=False))
    sql.add(Dense(4,activation='softmax'))
    nadam=keras.optimizers.Nadam()
    sql.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['categorical_accuracy'])
    ESt=EarlyStopping(patience=10,monitor='val_loss')
    RLP = keras.callbacks.ReduceLROnPlateau(patience=4, min_lr=0.00000001)
    print(sql.summary())
    print('Train...')
    sql.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_vail, y_vail),callbacks=[ESt,RLP])
    return sql




def load_data(data_name):
    if data_name=='12mfcc':
        traindata = np.array(pd.read_csv('F:\data/12mfcc/12train.csv'))
        vailddata = np.array(pd.read_csv('F:\data/12mfcc/12vaild.csv'))
        testdata = np.array(pd.read_csv('F:\data/12mfcc/12test.csv'))
        return traindata,vailddata,testdata


def preprocess(traindata,vailddata,testdata):

    x_train = traindata[:, 0:-1]
    y_train= traindata[:, -1]
    x_traindata_one = np.reshape(x_train, [-1, squlength, datadim])
    y_traindata_one = y_train[::squlength]
    y_traindata_oneer = keras.utils.to_categorical(y_traindata_one, numclass)

    x_test=testdata[:,0:-1]
    y_test=testdata[:,-1]
    x_testdata_one = np.reshape(x_test, [-1, squlength, datadim])
    y_testdata_one = y_test[::squlength]
    y_testdata_oneer = keras.utils.to_categorical(y_testdata_one, numclass)


    ###vailddata
    x_vaild=vailddata[:,0:-1]
    y_vaild=vailddata[:,-1]
    x_vailddata_one = np.reshape(x_vaild, [-1, squlength, datadim])
    y_vailddata_one = y_vaild[::squlength]
    y_vailddata_oneer = keras.utils.to_categorical(y_vailddata_one, numclass)
    return x_traindata_one,y_traindata_oneer,x_vailddata_one,y_vailddata_oneer,x_testdata_one,y_testdata_oneer


if __name__ == '__main__':
    traindata, vailddata, testdata=load_data('12mfcc')
    path = 'D:\ZYJ\savep/m3.hd5f'
    squlength = 200  # 样本长度
    datadim = 12  # 样本维度数
    x_train, y_train, x_vaild, y_vaild, x_test, y_test = preprocess(traindata, vailddata, testdata)
    model = creatmodel(x_train=x_train, y_train=y_train, x_vail=x_vaild,
                          y_vail=y_vaild)
    score, acc = model.evaluate(x_test, y_test)
    print(acc)







# traindataone = traindata
# vaildataone = vailddata
# x_testdataone = testdata
#
# traindataone = np.concatenate((traindataone[0:]))
# x_trainone = traindataone[:, 0:-1]
# y_trainone = traindataone[:, -1]
# dimtrain = x_trainone.shape[1]
# x_traindata_one = np.reshape(x_trainone, [-1, squlength, dimtrain])
# y_traindata_one = y_trainone[::squlength]
# y_traindata_oneer = keras.utils.to_categorical(y_traindata_one, numclass)
#
# # vaildata
# vaildataone = np.concatenate((vaildataone[0:]))
# x_vailone = vaildataone[:, 0:-1]
# y_vailone = vaildataone[:, -1]
# x_vaildata_one = np.reshape(x_vailone, [-1, squlength, dimtrain])
# y_vaildata_one = y_vailone[::squlength]
# y_vaildata_oneer = keras.utils.to_categorical(y_vaildata_one, numclass)
#

#
# # testdata
#
# x_testdataone = np.concatenate((x_testdataone[0:]))
# x_testone = x_testdataone[:, 0:-1]
# y_testone = x_testdataone[:, -1]
# x_testdata_one = np.reshape(x_testone, [-1, squlength, dimtrain])
# y_testdata_one = y_testone[::squlength]
# y_testdata_oneer = keras.utils.to_categorical(y_testdata_one, numclass)
#
# # test model
# scoreone, accone = modelone.evaluate(x_testdata_one, y_testdata_oneer)
# print(scoreone)
# print(accone)
# xone=modelone.predict(x_testdata_one)
# y_predone=np.argmax(xone,axis=1)




