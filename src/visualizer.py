import datetime
import pandas
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from pandas.plotting import lag_plot
from scipy import signal

'''
时间序列可视化模块
'''

# 以给定的labels，使用matplotlib绘制出（x,y）数据
def yearlyPlot(ySeries,year,month,day,plotName ="Plot",yAxisName="yData"):

	date = datetime.date(year,month,day)
	dateList = []
	for x in range(len(ySeries)):
		dateList.append(date+datetime.timedelta(days=x))

	plt.plot_date(x=dateList,y=ySeries,fmt="r-")
	plt.title(plotName)
	plt.ylabel(yAxisName)
	plt.xlabel("Date")
	plt.grid(True)
	plt.show()

# 针对 ySeries 的不同时间滞后绘制自相关因子
def autoCorrPlot(ySeries,plotName="plot"):
	plt.figure()
	plt.title(plotName)
	data = pandas.Series(ySeries)
	autocorrelation_plot(data)
	plt.show()

# 显示滞后图以确定时间序列数据是否非随机
def lagPlot(ySeries,plotName="plot"):
	plt.figure()
	plt.title(plotName)
	data = pandas.Series(ySeries)
	lag_plot(data, marker='2', c='green')
	plt.show()

# 显示给定时间序列的周期图 <ySeries>
def periodogramPlot(ySeries,plotName="Plot",xAxisName="Frequency",yAxisName="Frequency Strength"):
	trans = signal.periodogram(ySeries)
	plt.title(plotName)
	plt.xlabel(xAxisName)
	plt.ylabel(yAxisName)
	plt.plot(trans[0], trans[1], color='green')
	plt.show()

# 绘制对比图
def comparisonPlot(year,month,day,seriesList,nameList,plotName="Comparison of Values over Time", yAxisName="Predicted"):
	date = datetime.date(year,month,day)
	dateList = []
	for x in range(len(seriesList[0])):
		dateList.append(date+datetime.timedelta(days=x))
	colors = ["b","g","r","c","m","y","k","w"]
	currColor = 0
	legendVars = []
	for i in range(len(seriesList)):
		x, = plt.plot_date(x=dateList,y=seriesList[i],color=colors[currColor],linestyle="-",marker=".")
		legendVars.append(x)
		currColor += 1
		if (currColor >= len(colors)):
			currColor = 0
	plt.legend(legendVars, nameList)
	plt.title(plotName)
	plt.ylabel(yAxisName)
	plt.xlabel("Date")
	plt.show()