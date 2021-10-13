#建立GRU模型 训练
from math import log,exp
from datagen import constructData,loadSeries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statistics
import matplotlib.pyplot as plt
from sklearn import preprocessing
#建立GRU模型 训练
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import GRU
def expand_fea(data,sequence_length):
    result = []
    for index in range(len(data) - sequence_length):  # 循环170-5次
        result.append(data[index: index + sequence_length])  # 第i行到i+5
    result = np.array(result)  # 得到161个样本，样本形式为8天*3特征
    return result
files = ["data/elia_new/ELIA_LOAD_2008.xls","data/elia_new/ELIA_LOAD_2009.xls","data/elia_new/ELIA_LOAD_2010.xls",
  "data/elia_new/ELIA_LOAD_2011.xls","data/elia_new/ELIA_LOAD_2012.xls","data/elia_new/ELIA_LOAD_2013.xls",
  "data/elia_new/ELIA_LOAD_2014.xls","data/elia_new/ELIA_LOAD_2015.xls","data/elia_new/ELIA_LOAD_2016.xls",
           "data/elia_new/ELIA_LOAD_2017.xls"]
#建立、训练模型过程
def gru_model():
    # 构造输入数据
    df1 = pd.DataFrame(loadSeries(files))
    df1 = df1.iloc[:, :]
    # 数据归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    df0 = min_max_scaler.fit_transform(df1)
    df = pd.DataFrame(df0, columns=df1.columns)
    X = df.iloc[:-1, :]
    y = df.iloc[1:, -1]
    # 构造训练集测试集
    y = pd.DataFrame(y.values, columns=['target'])
    x = X
    input_size = len(x.iloc[1, :])

    # 设置LSTM、GRU的时间窗
    window = 7
    # 处理LSTM数据
    stock = df
    seq_len = window
    amount_of_features = len(stock.columns)  # 有几列
    data = stock.as_matrix()  # pd.DataFrame(stock) 表格转化为矩阵
    sequence_length = seq_len + 1  # 序列长度5+1
    result = expand_fea(data, sequence_length)
    row = round(0.9 * result.shape[0])  # 划分训练集测试集
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]
    # reshape成 5天*3特征

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    d = 0.01
    model = Sequential()#建立层次模型
    model.add(GRU(64, input_shape=(window, input_size), return_sequences=True))#建立LSTM层
    model.add(Dropout(d))#建立的遗忘层
    model.add(GRU(16, input_shape=(window, input_size), return_sequences=False))#建立LSTM层
    model.add(Dropout(d))#建立的遗忘层
    model.add(Dense(4,init='uniform',activation='relu'))   #建立全连接层
    model.add(Dense(1,init='uniform',activation='relu'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch =100, batch_size = 256) #训练模型nb_epoch次

    #在训练集上的拟合结果
    y_train_predict=model.predict(X_train)
    y_train_predict=y_train_predict[:,0]
    draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
    draw.iloc[100:150,0].plot(figsize=(12,6))
    draw.iloc[100:150,1].plot(figsize=(12,6))
    plt.legend(('real', 'predict'),fontsize='15')
    plt.title("Train Data",fontsize='30') #添加标题
    #展示在训练集上的表现

    #在测试集上的预测
    y_test_predict=model.predict(X_test)
    y_test_predict=y_test_predict[:,0]
    draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
    draw.iloc[:,0].plot(figsize=(12,6))
    draw.iloc[:,1].plot(figsize=(12,6))
    plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
    plt.title("Test Data",fontsize='30') #添加标题
    plt.show()
    #展示在测试集上的表现


    #输出结果
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    import math
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
    print('训练集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_train_predict, y_train))
    print(mean_squared_error(y_train_predict, y_train) )
    print(mape(y_train_predict, y_train) )
    print('测试集上的MAE/MSE/MAPE')
    print(mean_absolute_error(y_test_predict, y_test))
    print(mean_squared_error(y_test_predict, y_test) )
    print(mape(y_test_predict,  y_test) )
    y_var_test=y_test[1:]-y_test[:len(y_test)-1]
    y_var_predict=y_test_predict[1:]-y_test_predict[:len(y_test_predict)-1]
    txt=np.zeros(len(y_var_test))
    for i in range(len(y_var_test-1)):
        txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
    result=sum(txt)/len(txt)
    print('预测涨跌正确:',result)
if __name__=="__main__":
    gru_model()