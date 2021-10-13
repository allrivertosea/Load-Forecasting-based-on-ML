#建立bp模型 训练
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from datagen import constructData,loadSeries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statistics
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

data = constructData()
cutoff = len(data) - 364
data_X = minmax_scale(data[0])
data_Y = minmax_scale(data[1])
xTrain = data_X[0:cutoff]
yTrain = data_Y[0:cutoff]
xTest = data_X[cutoff:]
yTest = data_Y[cutoff:]
x = pd.DataFrame(data_X)
statistics.estimateMissing(xTrain, 0.0)
statistics.estimateMissing(xTest, 0.0)
input_size = len(x.iloc[1, :])

model = Sequential()  #层次模型
model.add(Dense(16,input_dim=input_size,init='uniform')) #输入层，Dense表示BP层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(4,init='uniform')) #中间层
model.add(Activation('relu'))  #添加激活函数
model.add(Dense(1))  #输出层
model.compile(loss='mean_squared_error', optimizer='Adam') #编译模型
model.fit(xTrain, yTrain, nb_epoch = 50, batch_size = 256) #训练模型nb_epoch=50次

# #在训练集上的拟合结果
y_train_predict=model.predict(xTrain)
y_train_predict=y_train_predict[:,0]
# draw=pd.concat([pd.DataFrame(yTrain),pd.DataFrame(y_train_predict)],axis=1)
# draw.iloc[100:150,0].plot(figsize=(12,6))
# draw.iloc[100:150,1].plot(figsize=(12,6))
# plt.legend(('real', 'predict'),fontsize='15')
# plt.title("Train Data",fontsize='30') #添加标题
#展示在训练集上的表现

#在测试集上的预测
y_test_predict=model.predict(xTest)
y_test_predict=y_test_predict[:,0]
# draw=pd.concat([pd.DataFrame(yTest),pd.DataFrame(y_test_predict)],axis=1);
# draw.iloc[:,0].plot(figsize=(12,6))
# draw.iloc[:,1].plot(figsize=(12,6))
# plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
# plt.title("Test Data",fontsize='30') #添加标题
#展示在测试集上的表现

#输出结果
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
print('训练集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_train_predict, yTrain))
print(mean_squared_error(y_train_predict, yTrain) )
# print(mape(y_train_predict, yTrain[:,0]) )
print('测试集上的MAE/MSE/MAPE')
print(mean_absolute_error(y_test_predict, yTest))
print(mean_squared_error(y_test_predict, yTest) )
# print(mape(y_test_predict,  yTest[:,0]) )
y_var_test=yTest[1:]-yTest[:len(yTest)-1]
y_var_predict=y_test_predict[1:]-y_test_predict[:len(y_test_predict)-1]
txt=np.zeros(len(y_var_test))
for i in range(len(y_var_test-1)):
    txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
result=sum(txt)/len(txt)
print('预测涨跌正确:',result)