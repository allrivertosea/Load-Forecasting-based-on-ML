import math
import visualizer
import statistics
import numpy as np
from datagen import constructData
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF,ConstantKernel as C

def gaussianProcesses():

    kernal =  C(1.0, (1e-3, 1e3)) * RBF(10, (0.5, 2))
    preds = []
    names = ['squared_exponential']


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

    # 使用 GPR 进行训练和测试
    pred = gaussProcPred(xTrain,yTrain,xTest,kernal)
    trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
    trendedPred = [math.exp(x) for x in trendedPred]
    #计算NRMSE
    err = statistics.normRmse(yTest,trendedPred)
    preds.append(trendedPred)
    preds.append(yTest)
    names.append("actual")
    print ("The Normalized Root-Mean Square Error is " + str(err) + " using covariance function " + "RBF" + "...")
    visualizer.comparisonPlot(2017, 1, 1, preds, names,
                              plotName="Gaussian Process Regression Load Predictions vs. Actual",
                              yAxisName="Predicted Kilowatts")


# 高斯过程回归
def gaussProcPred(xTrain,yTrain,xTest,covar):
    xTrainAlter = []
    for i in range(1,len(xTrain)):
        tvec = xTrain[i-1]+xTrain[i]
        xTrainAlter.append(tvec)
    xTestAlter = []
    xTestAlter.append(xTrain[len(xTrain)-1]+xTest[0])
    for i in range(1,len(xTest)):
        tvec = xTest[i-1]+xTest[i]
        xTestAlter.append(tvec)
    clfr = gaussian_process.GaussianProcessRegressor(kernel=covar)
    clfr.fit(xTrainAlter,yTrain[1:])
    return clfr.predict(xTestAlter)

if __name__=="__main__":
    gaussianProcesses()