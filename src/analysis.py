#! /usr/bin/python

import visualizer
import statistics
import numpy as np
from datagen import constructData

'''
负荷时间序列构成的多种统计意义的图片
'''

# 绘制原始负荷序列
def plotOriginal():
    data = constructData()
    # 过去 9 年的总电力需求图
    section = data[1][0:len(data[1])-365]
    visualizer.yearlyPlot(section,
        2008,1,1,"Average Total Electricity Load : 2008-2016","Kilowatts")

# 绘制用最小二乘法去除趋势后负荷序列
def plotDetrended():
    data = constructData()
    indices = np.arange(len(data[1]))
    detrendY = statistics.detrend(indices,data[1])[0]
    visualizer.yearlyPlot(detrendY,
        2008,1,1,"Detrended Aggregate Electricity Demand","Residual Kilowatts")

# 绘制负荷序列的相关图
# - 针对时间滞后绘制自回归相关系数
def plotCorrelogram():
    data = constructData()
    visualizer.autoCorrPlot(data[1][len(data[1])-730:len(data[1])-365],"Average Total Electricity Load Autocorrelations : 2016")

# 绘制负荷序列的滞后图
# - 用于判断时间序列数据是否非随机
def plotLag():
    data = constructData()
    visualizer.lagPlot(data[1][0:len(data[1])-365],"Average Total Electricity Load Lag : 2008-2016")
        
# 绘制负荷序列的周期图

def plotPeriodogram():
    data = constructData()
    visualizer.periodogramPlot(data[1][len(data[1])-730:len(data[1])-365],
        "Periodogram of Average Total Electricity Load : 2016")

# 绘制原始负荷序列与 VS 去趋势负荷序列
def plotOrigVsDetrend():
    data = constructData()
    # Original time series
    data1 = constructData()
    origY = data1[1][0:len(data[1])-365]
    # Detrended time series
    indices = np.arange(len(data[1])-365)
    detrendY = statistics.detrend(indices,data[1][0:len(data[1])-365])[0]

    visualizer.comparisonPlot(2008,1,1,origY,detrendY,plotName="Aggregate Electric Load : Original & Detrended", yAxisName="Kilowatts")

if __name__=="__main__":
    plotPeriodogram()