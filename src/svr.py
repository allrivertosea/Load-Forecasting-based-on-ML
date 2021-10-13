#! /usr/bin/python

import math
import statistics
import visualizer
import numpy as np
from datagen import constructData
from sklearn import svm
import warnings

def suppVectorRegress():

    kernelList = ["linear","rbf",polyKernel]
    names = ["linear","radial basis","poly"]
    preds = []

    # 索引时间序列数据并进行预处理
    data = constructData()


    cutoff = len(data)-364
    xTrain = data[0][0:cutoff]
    yTrain = data[1][0:cutoff]
    xTest = data[0][cutoff:]
    yTest = data[1][cutoff:]

    statistics.estimateMissing(xTrain,0.0)
    statistics.estimateMissing(xTest,0.0)

    xTrain = [[math.log(y) for y in x] for x in xTrain]
    xTest = [[math.log(y) for y in x] for x in xTest]
    yTrain = [math.log(x) for x in yTrain]

    indices = np.arange(len(data[1]))
    trainIndices = indices[0:cutoff]
    testIndices = indices[cutoff:]
    detrended,slope,intercept = statistics.detrend(trainIndices,yTrain)
    yTrain = detrended

    #免于告警
    warnings.filterwarnings("ignore")

    for gen in range(len(kernelList)):

        # 使用SVR进行训练和测试
        pred = svrPredictions(xTrain,yTrain,xTest,kernelList[gen])
        # 将趋势重新添加到预测中
        trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
        # 逆标准化
        trendedPred = [math.exp(x) for x in trendedPred]
        #计算NRMSE
        err = statistics.normRmse(yTest,trendedPred)

        print ("The Normalized Root-Mean Square Error is " + str(err) + " using kernel " + names[gen] + "...")

        preds.append(trendedPred)

    names.append("actual")
    preds.append(yTest)

    visualizer.comparisonPlot(2017,1,1,preds,names,plotName="Support Vector Regression Load Predictions vs. Actual",
        yAxisName="Predicted Kilowatts")

# 构建一个支持向量机并得到测试集的预测，返回一个一维的预测向量
def svrPredictions(xTrain,yTrain,xTest,k):
    clf = svm.SVR(C=2.0,kernel=k)
    clf.fit(xTrain,yTrain)
    return clf.predict(xTest)

# 多项式核函数
def polyKernel(x,y):
    return (np.dot(x,y.T)+1.0)**0.95

if __name__=="__main__":
    suppVectorRegress()
