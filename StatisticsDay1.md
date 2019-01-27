#### 统计学习基本方法

统计学习（statistical learning）是关于计算机基于数据构建概率统计模型并运用模型对数据进行预测和分析的一门学科。统计学习也称为统计机器学习（statistial machine learning）。

**1.统计学习的主要特点是：**

1. 统计学习以计算机及网络为平台，是建立在计算机及网络之上的；
2. 统计学习以数据为研究对象，是数据驱动的学科；
3. 统计学习的目的是对数据进行预测与分析；
4. 统计学习以方法为中心，统计学习方法构建模型并应用模型进行预测与分析；
5. 统计学习是概率论、统计学、信息论、计算理论、最优化理论及计算机科学等多个领域的交叉学科，并且在发展中逐步形成独自的理论体系与方法论。

**2.统计学习的对象**

统计学习的对象是数据（data），它从数据出发，提取数据的特征，抽象出数据的模型，发现数据中的知识，又回到对数据的分析与预测中去。数据可包括数字文字、图像、视频、音频等。

统计学习关于数据的基本假设是同类数据具有一定的统计规律性，这是统计学习的前提；如用随机变量描述数据中的特征，用概率分布描述数据的统计规律。

**3.统计学习的目的**

统计学习用于对数据进行预测与分析，其是通过构建概率统计模型实现的。

**4.统计学习的方法**

统计学习的方法是基于数据构建统计模型从而对数据进行预测与分析，统计学习由监督学习（supervised learning）、非监督学习（unsupervised learning）、半监督学习（semi-supervised learning）、强化学习（reinforcement learning）等组成。

统计学习方法三要素：模型（model）+策略（strategy）+算法（algorithm）

实现统计学习方法的步骤如下：

1. 得到一个有限的训练数据集合
2. 确定包含所有可能的模型的假设空间，即学习模型的集合
3. 确定模型选择的准则，即学习的策略
4. 实现求解最优模型的算法，即学习的算法
5. 通过学习方法选择最优模型
6. 利用学习的最优模型对新数据进行预测或分析。

**5.统计学习的研究**

统计学习研究一般包括统计学习方法（statistical learning method）、统计学习理论（statistical learning theory）、统计学习应用（application of statistical learning）。

#### 统计学习的基本概念

1. **总体（population）**：根据研究目的确定的同类对象的全体（集合）：样本（sample）：从总体中随机抽取的部分具有代表性的研究对象。

2. **参数（Parameter）**：反映总体特征的统计指标，如总体均数、标准差等，是固定的常量。

3. **统计量（statistic）**：反映样本特征的统计指标，如样本均数、标准差等，是在参数附近波动的随机变量。

4. **统计资料分布（statistical distribution）**：定量（计量）资料、定性（计数）资料、等级资料。

   **计量资料统计描述**：

   - 集中趋势：均数（mean）、中位数（median）、众数（mode）。
   - 离散趋势：极差（range）、四分位间距（interquartile range）(QR=P75-P25$)、标准差(standard deviation)（或方差(variance））、变异系数（variable coefficient）。

5. **二项分布（Bionomial Distribution）**：

   说起二项分布(binomial distribution)，不得不提的前提是伯努利试验(Bernoulli experiment)，也即n次独立重复试验。伯努利试验是在同样的条件下重复、相互独立进行的一种随机试验。

      伯努利试验的特点是：

   （1）每次试验中事件只有两种结果：事件发生或者不发生，如硬币正面或反面，患病或没患病；

   （2）每次试验中事件发生的概率是相同的，注意不一定是0.5；

   （3）n次试验的事件相互之间独立。

   举个实例，最简单的抛硬币试验就是伯努利试验，在一次试验中硬币要么正面朝上，要么反面朝上，每次正面朝上的概率都一样p=0.5，且每次抛硬币的事件相互独立，即每次正面朝上的概率不受其他试验的影响。如果独立重复抛n=10次硬币，正面朝上的次数k可能为0,1,2,3,4,5,6,7,8,9,10中的任何一个，那么k显然是一个随机变量，这里就称随机变量k服从二项分布。

   n次抛硬币中恰好出现k次的概率为：
   $$
   P(X=k) = C(n,k) * pk*(1-p)^{(n-k)}
   $$

这就是二项分布的分布律，记作X~B(n,p)，其中C(n,k)是组合数，在数学中也叫二项式系数，这就是二项分布名称的来历。判断某个随机变量X是否符合二项分布除了满足上述的伯努利试验外，关键是这个X是否表示事件发生的次数。二项分布的数学期望E(X)=n*p，方差D(X)=n*p*(1-p)。

6. **泊松分布（poisson distribution）**

   泊松分布描述的是一个离散随机事件在单位时间内发生的次数, 其对应的场景是我们统计已知单位事件内发生某事件的平均次数**λ**, 那么我们在一个单位事件内发生**k**次的概率是多大呢? 

   ​        比如说医院产房里统计历史数据可知, 平均每小时出生3个宝宝,那么在接下来的一个小时内, 出生 0 个宝宝, 1 个宝宝, …, 3 个宝宝, …10 个宝宝, n 个宝宝的概率分别是多少呢? 泊松分布给出了定量的结果 :  
   $$
   P(X = k) = \frac {\lambda^k}{k!}e^{-\lambda}, k = 0,1,2...n
   $$
    其中 $$P(X=k) $$描述的就是在单位时间内事件 *X*发生 *k*次的概率,  *λ*代表在单位时间内事件发生的平均次数, 也就是泊松分布的期望, 同时也是方差. 

   一个场景可以用泊松分布来描述, 需要满足三个条件：

   1. 均值稳定. 即 *λ*在任意划定的单位时间长度内,应该是一个稳定的数值.

   2. 事件独立. 事件之间相互独立, 若相关, 则泊松分布失效.

   3. 在一个极小的时间内, 事件发生的次数应趋近于0. 比如说：产房平均 1 小时出生 3 个宝宝, 那我任意指定1ms, 那这1ms 内出生的宝宝数趋近于 0 .

7. **大数定理（Law of large numbers)**

   设随机变量$X_1,X_2,X_3,...,X_n$是一列互相独立的随机变量（或者两两不相关），并且分别存在期望$E(X_k)$,则对于任意小的正数ε有：
   $$
   \lim_{x→∞}P(|\frac{1}{n}\sum_{k=1}^nX_k-\frac{1}{n}\sum_{k=1}^nE(X_k)|<ε)
   $$
   理解：随着样本数量n的增加，样本的平均数（总体中的一部分）将接近于总体样本的平均数，所以在统计推断中一般使用样本平均数估计总体平均数的值。

   下面我们用简单的python代码来客观展现出大数定理：

```PYTHON
####https://blog.csdn.net/qq_26347025/article/details/78957001###
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
# 防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
 
# 定义数据量大小
numberSize = 200
# 生成服从正态分布的随机数据，其均值为0
randData = np.random.normal(scale=100, size=numberSize)
# 保存随机每增加一个数据后算出来的均值
randData_average = []
# 保存每增加一个数据后的数据和
randData_sum = 0
# 通过循环计算每增加一个数据后的均值
for index in range(len(randData)):
    randData_average.append((randData[index] + randData_sum) / (index + 1.0))
# 定义作图的x值和y值
x = np.arange(0,numberSize,1)
y = randData_average
# 作图设置
plt.title('大数定律')
plt.xlabel('数据量')
plt.ylabel('平均值')
# 作图并展示
plt.plot(x,y)
plt.plot([0,numberSize], [0,0], 'r')
plt.show()
```

![1548590606482](C:\Users\18019\AppData\Roaming\Typora\typora-user-images\1548590606482.png)

由上图即可进一步验证我们的结论，随着数据量的增加，均值越来越接近实际均值0。

即当我们的样本数据量足够大的时候，我们就可以用样本的平均值来估计总体平均值。

8. **正态分布（Normal distribution）**：正态分布又叫高斯分布，若随机变量$X$服从一个数学期望为$μ$、方差为$σ^2$的正态分布，记为$N(μ，σ^2)$:

$$
{\displaystyle X\sim N(\mu ,\sigma ^{2})}
$$

其概率密度函数为正态分布的期望值$μ$决定了其位置，其标准差$σ$决定了分布的幅度。当$μ = 0,σ = 1$时的正态分布是标准正态分布：
$$
{\displaystyle f(x)={1 \over \sigma {\sqrt {2\pi }}}\,e^{-{(x-\mu )^{2} \over 2\sigma ^{2}}}}
$$

```PYTHON
###https://blog.csdn.net/bitcarmanlee/article/details/79153932###
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def demo1():
    mu, sigma, num_bins = 0,1,50  ##50为直方图的数量
    x = mu+sigma*np.random.randn(1000000)
    n, bins, patches = plt.hist(x, num_bins, normed = True,facecolor ='blue',alpha = 0.5) #直方图函数，x为x轴的值，normed=True正则化直方图，即让每个方条表示年龄在该区间内的数量占总数量的比，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
    y = mlab.normpdf(bins,mu,sigma) #拟合一条最佳正态分布曲线y
    plt.plot(bins,y,'r--')
    plt.xlabel('Expection')
    plt.ylabel('Probability')
    plt.title('Histogram of Normal Distribution:$\mu = 0$,$\sigma =1$')
    
    
    plt.subplots_adjust(left = 0.15)
    plt.show()
    
    
demo1()
```

![1548594212672](C:\Users\18019\AppData\Roaming\Typora\typora-user-images\1548594212672.png)