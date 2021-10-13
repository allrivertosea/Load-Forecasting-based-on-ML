import numpy as np # linear algebra
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datagen import constructData
from sklearn.preprocessing import minmax_scale
import statistics
import matplotlib.pyplot as plt

def xgd_reg():
    data = constructData()
    cutoff = len(data) - 364
    data_X = minmax_scale(data[0])
    data_Y = minmax_scale(data[1])
    xTrain = data_X[0:cutoff]
    yTrain = data_Y[0:cutoff]
    xTest = data_X[cutoff:]
    yTest = data_Y[cutoff:]
    statistics.estimateMissing(xTrain, 0.0)
    statistics.estimateMissing(xTest, 0.0)
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(xTrain, yTrain) # Change verbose to True if you want to see it train
    y_test_predict = reg.predict(xTest)
    # 展示在测试集上的表现
    draw = pd.concat([pd.DataFrame(yTest), pd.DataFrame(y_test_predict)], axis=1);
    draw.iloc[:, 0].plot(figsize=(12, 6))
    draw.iloc[:, 1].plot(figsize=(12, 6))
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Test Data", fontsize='30')  # 添加标题
    print(y_test_predict)
    print(yTest)

    # 输出结果
    print('测试集上的MAE/MSE/NRMSE')  # 结果不好看的缘故是因为逻辑回归需要进行归一化。
    print(mean_absolute_error(y_test_predict, yTest))
    print(mean_squared_error(y_test_predict, yTest))
    print(statistics.normRmse(yTest, y_test_predict))
    # print(mape(y_test_predict,  yTest[:,0]) )
    # 展示在测试集上的表现
    draw = pd.concat([pd.DataFrame(yTest), pd.DataFrame(y_test_predict)], axis=1)
    draw.iloc[:, 0].plot(figsize=(12, 6))
    draw.iloc[:, 1].plot(figsize=(12, 6))
    plt.legend(('real', 'predict'), loc='upper right', fontsize='15')
    plt.title("Test Data", fontsize='30')  # 添加标题
    plt.show()
if __name__=="__main__":
    xgd_reg()
