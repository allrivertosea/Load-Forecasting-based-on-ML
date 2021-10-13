
import math
import statistics
import visualizer
import numpy as np
from datagen import constructData
from scipy.spatial import distance
from scipy.cluster.vq import kmeans
from scipy.spatial.distance import euclidean
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import warnings

def clustering():

    # 索引时间序列数据并进行预处理
    data = constructData()

    # 2017 有 365 天，但我们取 364 天，因为最后一天没有数值
    cutoff = len(data)-364
    xTrain = data[0][0:cutoff]
    yTrain = data[1][0:cutoff]
    xTest = data[0][cutoff:]
    yTest = data[1][cutoff:]

    # 以两个邻居的平均值填充由零表示的缺失值
    statistics.estimateMissing(xTrain,0.0)
    statistics.estimateMissing(xTest,0.0)

    # 对数缩放数据
    xTrain = [[math.log(y) for y in x] for x in xTrain]
    xTest = [[math.log(y) for y in x] for x in xTest]
    yTrain = [math.log(x) for x in yTrain]

    # 时间序列去趋势
    indices = np.arange(len(data[1]))
    trainIndices = indices[0:cutoff]
    testIndices = indices[cutoff:]
    detrended,slope,intercept = statistics.detrend(trainIndices,yTrain)
    yTrain = detrended

    # 计算数据的簇心和标签,季节性周期有周7类，年365类
    cward_7,lward_7 = hierarchicalClustering(xTrain,7)
    cward_365,lward_365 = hierarchicalClustering(xTrain,365)

    ckmeans_7,lkmeans_7 = kMeansClustering(xTrain,7)
    ckmeans_365,lkmeans_365 = kMeansClustering(xTrain,365)

    c = [cward_7,cward_365,ckmeans_7,ckmeans_365]
    l = [lward_7,lward_365,lkmeans_7,lkmeans_365]

    algNames = ["agglomerative(7)","agglomerative(365)","k-means(7)","k-means(365)"]

    preds = []

    for t in range(len(c)):
        # 当前聚类算法计算的簇心
        centroids = c[t]
        # 当前聚类定义的样本的标签
        labels = l[t]

        # 将训练样本分成簇集
        clusterSets = []
        # 样本的时间标签，分成簇
        timeLabels = []

        for x in range(len(centroids)):
            clusterSets.append([])
        for x in range(len(labels)):
            #把样本放入它的簇内
            clusterSets[labels[x]].append((xTrain[x],yTrain[x]))
        # 计算每个测试样本的预测值
        pred = predictClustering(centroids,clusterSets,xTest,"euclidean")
        # 将趋势重新添加到预测值中
        trendedPred = statistics.reapplyTrend(testIndices,pred,slope,intercept)
        # 逆标准化

        trendedPred = [math.exp(x) for x in trendedPred]

        # 计算 NRMSE
        err = statistics.normRmse(yTest,trendedPred)
        # 将预测值添加到predictions列表中
        preds.append(trendedPred)

        print ("The Normalized Root-Mean Square Error is " + str(err) + " using algorithm " + algNames[t] + "...")

    algNames.append("actual")
    preds.append(yTest)

    visualizer.comparisonPlot(2017,1,1,preds,algNames,
        plotName="Clustering Load Predictions vs. Actual", 
        yAxisName="Predicted Kilowatts")
    warnings.filterwarnings("ignore")

# 实施使用ward方法的凝聚层次聚类
def hierarchicalClustering(x,k):
    model = AgglomerativeClustering(n_clusters=k,linkage="ward")
    labels = model.fit_predict(np.asarray(x))

    # Centroids是一个列表的列表
    centroids = []
    for c in range(k):
        base = []
        for d in range(len(x[0])):
            base.append(0)
        centroids.append(base)

    # 每个簇存储一定数目的样本
    ctrs = np.zeros(k)

    # 加和每个集群的所有向量
    for c in range(len(x)):
        centDex = labels[c]
        for d in range(len(centroids[centDex])):
            centroids[centDex][d] += x[c][d]
        ctrs[centDex] += 1

    # 计算每个集群中向量的平均值，来得到簇心
    for c in range(len(centroids)):
        for d in range(len(centroids[c])):
            centroids[c][d] = centroids[c][d]/ctrs[c]

    return (centroids,labels)


'''
对带有参数 k 的向量 x 的有序序列执行 K 均值聚类，并返回一个 2 元组： 第一个元组值是质心列表 
第二个元组值是长度等于 x 的向量 x'，使得x' 的第 i 个值是x 的第 i 个样本的聚类标签
'''
def kMeansClustering(x,k):

    # 将列表转换为 numpy 格式
    conv = np.asarray(x)

    # 计算簇心
    centroids = kmeans(conv,k,iter=10)[0]

    # 重新标记x
    labels = []
    for y in range(len(x)):
        minDist = float('inf')
        minLabel = -1
        for z in range(len(centroids)):
            e = euclidean(conv[y],centroids[z])
            if (e < minDist):
                minDist = e
                minLabel = z
        labels.append(minLabel)

    # 返回簇心和标签列表
    return (centroids,labels)

# 对 xTest 中的示例执行加权聚类
# 返回 predictions的一维向量
def predictClustering(clusters,clusterSets,xTest,metric):
    clustLabels = []
    simFunction = getDistLambda(metric)
    for x in range(len(xTest)):
        clustDex = -1
        clustDist = float('inf')
        for y in range(len(clusters)):
            dist = simFunction(clusters[y],xTest[x])
            if (dist < clustDist):
                clustDist = dist
                clustDex = y
        clustLabels.append(clustDex)
    predict = np.zeros(len(xTest))
    for x in range(len(xTest)):
        predict[x] = weightedClusterClass(xTest[x],clusterSets[clustLabels[x]],simFunction)
    return predict

# 实施加权聚类分类
def weightedClusterClass(xVector,examples,simFunction):
    pred = 0.0
    normalizer = 0.0
    ctr = 0
    for x in examples:
        similarity = 1.0/simFunction(xVector,x[0])
        pred += similarity*x[1]
        normalizer += similarity
        ctr += 1
    return (pred/normalizer)

def getDistLambda(metric):
    if (metric == "manhattan"):
        return lambda x,y : distance.cityblock(x,y)
    elif (metric == "cosine"):
        return lambda x,y : distance.cosine(x,y)
    else:
        return lambda x,y : distance.euclidean(x,y)

if __name__=="__main__":
    clustering()