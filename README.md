# Load-Forecasting-based-on-ML
# 基于机器学习的负荷预测

该项目使用来自 Elia 的免费可用负荷时间序列数据，目的是开发出准确的方法来提前一天预测 Elia 电网的平均总负荷。 已经开发了十一种不同的基于机器学习的算法，每种算法都可以通过运行 src 文件夹中的相应脚本来运行。 各脚本如下

* 逻辑回归 - lr.py
* 基于Sigmoid的神经网络 - neural.py
* 加权聚类 - clustering.py
* 支持向量回归 - svr.py
* CART回归树 - cart.py
* 极限梯度提升树 - xgboost.py
* 随机森林 - randforest.py
* 长短时记忆神经网络 - lstm.py
* 门控递归单元神经网络 - gru.py
* 自回归差分移动平均 - arima.py
* 高斯过程回归 - gpr.py

脚本所需的数据存储在 src 内的 data 文件夹中。 脚本 analysis.py 提供了可视化 Elia 负荷时间序列各个方面的函数。使用 Visualizer.py 中的方法显示结果预测。 

**注意**

有关Elia数据集、算法开发和预测结果的更多详细信息，请查看 writeup 文件夹中的 PDF文件：机器学习预测电网平均总负荷。

## 数据来源:

* Elia 提供的电力负荷数据集.
* 链接 : http://www.elia.be/en/grid-data
