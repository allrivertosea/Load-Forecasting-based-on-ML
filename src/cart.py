from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
import math
from datagen import constructData
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statistics
from sklearn.preprocessing import minmax_scale
from sklearn.ensemble import RandomForestRegressor
import visualizer
def cart():
    names = ["cart"]
    preds = []
    data = constructData()
    cutoff = len(data) - 364
    # data_X = minmax_scale(data[0])
    # data_Y = minmax_scale(data[1])

    xTrain = data[0][0:cutoff]
    yTrain = data[1][0:cutoff]
    xTest = data[0][cutoff:]
    yTest = data[1][cutoff:]

    statistics.estimateMissing(xTrain, 0.0)
    statistics.estimateMissing(xTest, 0.0)

    xTrain = [[math.log(y) for y in x] for x in xTrain]
    xTest = [[math.log(y) for y in x] for x in xTest]
    yTrain = [math.log(x) for x in yTrain]

    indices = np.arange(len(data[1]))
    trainIndices = indices[0:cutoff]
    testIndices = indices[cutoff:]
    detrended, slope, intercept = statistics.detrend(trainIndices, yTrain)
    yTrain = detrended


    model = DecisionTreeRegressor(max_depth=8, max_leaf_nodes=16)#该部分需要调参
    model.fit(xTrain, yTrain)
    #在测试集上的预测
    pred=model.predict(xTest)
    # 将趋势重新添加到预测中
    trendedPred = statistics.reapplyTrend(testIndices, pred, slope, intercept)
    # 逆标准化
    trendedPred = [math.exp(x) for x in trendedPred]
    # 计算NRMSE
    preds.append(trendedPred)
    preds.append(yTest)
    err = statistics.normRmse(yTest, trendedPred)
    print("The Normalized Root-Mean Square Error is " + str(err) + " cart " + names[0] + "...")
    names.append("actual")
    visualizer.comparisonPlot(2017, 1, 1, preds, names,
                              plotName="Cart vs. Actual",
                              yAxisName="Predicted Kilowatts")

if __name__=="__main__":
    cart()
