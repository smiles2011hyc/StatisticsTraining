#### 假设检验

​        **假设检验（hypothesis tseting)**，或者叫做**显著性检验（significance testing)**是数理统计学中根据一定假设条件由样本推断总体的一种方法。其原理是先对总体的特征做出某种假设，然后通过抽样研究的统计推理，对此假设应该被拒绝还是接受推断。由于是以假设为前提，故而需要进行的假设为：

​    $H_0$:原假设或零假设（null hypothesis），即需要去验证的假设；一般首先认定原假设是正确的，然后根据显著性水平选择是接受还是拒绝原假设。

​    $H_1$:备择假设（alternative hypothesis），一般是原假设的否命题，当原假设被拒绝时，默认接受备择假设。

​    如原假设时假设总体均值$u=u_0$,则备择假设为总体均值$μ≠μ_0$,检验的过程就是计算$H_0$正确的情况下相应的统计量和显著性概率，来验证$H_0$应该被接受还是拒绝。

如：一个神经学家注射100个小鼠药物后的反应时间，当一单位剂量的药物注射后，记录其反应时间。神经学家知道当未注射药物后的反应时间为1.2s，注射后的100个小鼠反应时间为1.05s，其标准差为0.5s，你认为这个药物是否对小鼠的反应时间有影响？

​        $H_0$:没有影响 $u=1.2s$

​        $H_1$：有影响 $u\neq1.2s$

```python
from scipy import stats
import numpy as np
#standard deviation
DS = 0.5
mean1 = 1.2
mean2 = 1.05
N = 100
Z = (mean1 - mean2)/(DS/np.sqrt(N))
df = 100
p = 1 - stats.t.cdf(t,df = df)
print("Z = ", Z )
print("p = ", 2*p)
if p < 0.05:
    print("由于p<0.05,故而null hypothesis不成立，拒绝null hypothesis，则有影响")
else:
    print("由于p>0.05,故而null hypothesis成立，接受null hypothesis，则无影响")
#Z =  2.9999999999999982
#p =  0.003407915343329515
#由于p<0.05,故而null hypothesis不成立，拒绝null hypothesis，则有影响
```

上例仅仅说明了药物有影响，但是没有说明药物是积极还是消极影响，这在统计学上称其为**双侧检验（a two-tailed test)**；如果需要考虑这些因素，即由于药物主要是减少反应时间，则仅考虑这一个结果，则在统计学上称为**单侧检验（one-tailed test)**：

​        $H_0​$：没有影响 $u=1.2s​$

​        $H_1$：有药物减少了小鼠反应影响 $u<1.2s。$

由于双侧检验的$p=0.003$,故而单侧的面积应为$p$的一半，故而$H_0$不成立，又由于施加药物后平均值小于control的平均值，所以接受$H_1$。

#### Z-statistic 和t-statistic的区别

当样本容量很小时，样本均值抽样分布不应该采用正态分布，而应采用t分布。Z统计量服从正态分布（上例中$sample =100$），而t统计量服从t分布，根据经验给出了样本容量30的界限，经验上告诉人们如何在z统计量和t统计量之间进行取舍。

Example：环保标准规定汽车的新排放标准：平均值<20ppm，现某汽车公司测试10辆汽车的排放结果如下：15.6 16.2 22.5 20.5 16.4 19.4 16.6 17.9 12.7 13.9 。问题：公司引擎排放是否满足新标准？

​        $H_0$：$u=20ppm$，不满足新标准

​        $H_1$：$u<20ppm$，满足新标准

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataSet = pd.Series([15.6,16.2,22.5,20.5,16.4,19.4,16.6,17.9,12.7,13.9])
#样本平均值
sample_mean = dataSet.mean()
print("样本平均值是：", sample_mean)
#样本平均值是： 17.169999999999998
#样本标准差
sample_std = dataSet.std()
print("样本标准差是：", sample_std)
#样本标准差是： 2.9814426038413018
```

当$Sample >30$时，符合中心极限定理，抽样分布呈正态分布;当$Sample<30$时，抽样分布符合t分布或其他分布。可以使用python中的**displot绘图，画出直方图和拟合曲线。**

```python
import seaborn as sns
#解决画图中中文乱码，解决负号问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
#绘图
sns.distplot(dataSet)
plt.title('数据集分布')
plt.show()
```

![1548834363138](https://user-images.githubusercontent.com/15357935/51983187-014c7c80-24d3-11e9-9814-06db4c69839e.png)

```python
#排放标准
standard_mean = 20
t, p_twoTail = stats.ttest_1samp(dataSet,standard_mean)
p_oneTail = p_twoTail/2
print("t =",t,'\n'"p_twoTail=",p_twoTail,'\n'"p_oneTail = ", p_oneTail)
#犯Ⅰ型错误的概率低于0.01
alpha = 0.01
if (t<0 and p_oneTail < alpha):
    print("拒绝零假设，有统计显著性，也就是汽车引擎排放满足标准")
else:
    print("接受零假设，没有统计显著，也就是汽车引擎排放不满足标准")
#t = -3.001649525885985 
#p_twoTail= 0.014916414248897527 
#p_oneTail =  0.0074582071244487635
#拒绝零假设，有统计显著性，也就是汽车引擎排放满足标准
```

#### 统计学上的第Ⅰ和第Ⅱ类错误

![1548836690479](https://user-images.githubusercontent.com/15357935/51983191-04476d00-24d3-11e9-9b09-3707cda3e6ed.png)

- **Ⅰ型错误**：格子B反应的情况是，实际上不存在差异（也就是干预措施无效），但是我们的研究拒绝了零假设，这叫做**Ⅰ型（type Ⅰ）错误**。
- **Ⅱ型错误**：格子C反映了相反的情况——处理措施确实有效，但是我们的研究不拒绝$H_0$,统计学家称其为**Ⅱ型（typeⅡ）错误**。































































































































































































