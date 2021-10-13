#该程序实施ARIMA时间序列模型

from pandas import read_csv
from pandas import datetime
import statistics
import math
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from datagen import const_df,constructData
import numpy as np
import visualizer

def arima_model():
        names = ["arima"]
        preds = []
        # 索引时间序列数据并进行预处理
        data = constructData()
        cutoff = len(data) - 364
        train = data[1][0:cutoff]
        test = data[1][cutoff:]
        # 对数缩放数据
        yTrain = [math.log(x) for x in train]
        yTest = test
        yTest_temp = [math.log(x) for x in yTest]
        prediction = []
        print(yTrain)
        for t in range(len(yTest)):
                model = ARIMA(yTrain, order=(5,1,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                # print(output)
                prediction.append(output[0][0])
                obs = yTest_temp[t]
                yTrain.append(obs)
                print('time =',t)
                # print(prediction)
        # 逆标准化
        pred = [math.exp(x) for x in prediction]
        preds.append(pred)

        print(yTest)
        print(preds)
        err2 = statistics.normRmse(yTest,pred)
        print('Test Score: %.2f NRMSE' % (err2))
        preds.append(yTest)
        names.append('actual')

        # 绘图
        visualizer.comparisonPlot(2017, 1, 1, preds, names,
                                  plotName="Arima vs. Actual",
                                  yAxisName="Predicted Kilowatts")
if __name__=="__main__":
        arima_model()