import xlrd
import math
import statistics
import numpy as np
import pandas as pd

'''
Functions for retrieving Elia dataset 
& forming training/testing datasets
'''
#构造dataframe数据集
def const_df(data):
  data_xtem = data[0]
  data_ytem = data[1]
  data_x = pd.DataFrame(np.array(data_xtem))
  data_y = pd.DataFrame(np.array(data_ytem))
  new_coly =['96']
  data_y.columns =new_coly
  return data_x,data_y

# 构造用于模拟实验的数据集
def constructData():
  files = ["data/elia_new/ELIA_LOAD_2008.xls","data/elia_new/ELIA_LOAD_2009.xls","data/elia_new/ELIA_LOAD_2010.xls",
  "data/elia_new/ELIA_LOAD_2011.xls","data/elia_new/ELIA_LOAD_2012.xls","data/elia_new/ELIA_LOAD_2013.xls",
  "data/elia_new/ELIA_LOAD_2014.xls","data/elia_new/ELIA_LOAD_2015.xls","data/elia_new/ELIA_LOAD_2016.xls",
           "data/elia_new/ELIA_LOAD_2017.xls"]
  return labelSeries(loadSeries(files))

# 构造标签数据
def labelSeries(series):
  xData = []
  yData = []
  for x in range(len(series)-1):
    xData.append(series[x])
    yData.append(np.mean(series[x+1]))
  return (xData,yData)

# arg1：Elia excel 电子表格文件名列表
# returns : 负荷单变量时间序列
def loadSeries(fileList):
  # 索引时间序列样本
  xData = []
  for fileName in fileList:
    book = xlrd.open_workbook(fileName,encoding_override='cp1252')
    sheet = book.sheet_by_index(0)
    for rx in range(2,sheet.nrows):
      row = sheet.row(rx)[3:]
      row = [row[x].value for x in range(0,len(row)-4)]
      xData.append(row)
  return xData