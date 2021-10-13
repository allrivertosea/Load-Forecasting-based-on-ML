#! /usr/bin/python

import visualizer
import math
import statistics
import numpy as np

from datagen import constructData
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.decomposition import PCA

# 神经网络-负荷预测
def neuralNetwork():
  

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

  # 时间序列去趋势
  indices = np.arange(len(data[1]))
  trainIndices = indices[0:cutoff]
  testIndices = indices[cutoff:]
  detrended,slope,intercept = statistics.detrend(trainIndices,yTrain)
  yTrain = detrended

  dimensions = [6,10,12]
  neurons = [30,50,50]

  names = []
  for x in range(len(dimensions)):
    s = "d=" + str(dimensions[x]) + ",h=" + str(neurons[x])
    names.append(s)

  preds = []

  for x in range(len(dimensions)):
    # 对特征向量进行降维
    pca = PCA(n_components=dimensions[x])
    pca.fit(xTrain)
    xTrainRed = pca.transform(xTrain)
    xTestRed = pca.transform(xTest)

    pred = fit_predict(xTrainRed,yTrain,xTestRed,40,neurons[x])

    # 将趋势重新添加到预测中
    trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
    #逆标准化
    trendedPred = [math.exp(x) for x in trendedPred]
    # 计算 NRMSE
    err = statistics.normRmse(yTest,trendedPred)

    preds.append(trendedPred)

    print ("The NRMSE for the neural network is " + str(err) + "...")

  preds.append(yTest)
  names.append("actual")

  visualizer.comparisonPlot(2017,1,1,preds,names,plotName="Neural Network Load Predictions vs. Actual",
        yAxisName="Predicted Kilowatts")


'''
构建一个具有给定神经元数量的神经网络，并将其拟合到指定时期数的训练数据，
并返回给定测试数据的预测值向量 - 假设目标是单变量
'''

def fit_predict(xTrain,yTrain,xTest,epochs,neurons):

  # 检查边界情况
  if (not len(xTrain) == len(yTrain) or len(xTrain) == 0 or 
    len(xTest) == 0 or epochs <= 0):
    return

  # 随机化训练数据（可能没有必要，但 pybrain 可能不会对数据本身进行混洗，因此执行安全检查）
  indices = np.arange(len(xTrain))
  np.random.shuffle(indices)

  trainSwapX = [xTrain[x] for x in indices]
  trainSwapY = [yTrain[x] for x in indices]

  supTrain = SupervisedDataSet(len(xTrain[0]),1)
  for x in range(len(trainSwapX)):
    supTrain.addSample(trainSwapX[x],trainSwapY[x])

  # 构建前馈神经网络

  n = FeedForwardNetwork()

  inLayer = LinearLayer(len(xTrain[0]))
  hiddenLayer1 = SigmoidLayer(neurons)
  outLayer = LinearLayer(1)

  n.addInputModule(inLayer)
  n.addModule(hiddenLayer1)
  n.addOutputModule(outLayer)

  in_to_hidden = FullConnection(inLayer, hiddenLayer1)
  hidden_to_out = FullConnection(hiddenLayer1, outLayer)
  
  n.addConnection(in_to_hidden)
  n.addConnection(hidden_to_out)

  n.sortModules() 

  # 在训练集上训练神经网络，在验证集上验证训练进度

  trainer = BackpropTrainer(n,dataset=supTrain,momentum=0.1,learningrate=0.01
    ,verbose=False,weightdecay=0.01)
  
  trainer.trainUntilConvergence(dataset=supTrain,
    maxEpochs=epochs,validationProportion=0.30)

  outputs = []
  for x in xTest:
    outputs.append(n.activate(x))

  return outputs

if __name__ == "__main__":
  neuralNetwork()