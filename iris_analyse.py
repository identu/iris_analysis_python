import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(palette="muted", color_codes=True)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=1.2)  # 解决Seaborn中文显示问题

# 载入数据
iris_data = pd.read_csv('iris.csv')
# 显示数据
print('**********Display Data**********')
print(iris_data)
# 显示数据信息
print('\n**********Data Information**********')
iris_data.info()
# 数据统计
print('\n**********Data Statistic**********')
print(iris_data.describe())
# 检查重复项
print('\n**********Checking For Duplicate Entries**********')
print(iris_data[iris_data.duplicated()])
# 检查数据集平衡性
print('\n**********Checking the balance**********')
print(iris_data['variety'].value_counts())
# 每个物种的平均值和中值
print(iris_data.groupby('variety').agg(['mean', 'median']))

# 数据可视化
print('\n**********Data Visualization**********')
print('Generating...')
# 每个物种的数量
plt.title('Species Count')
sns.histplot(iris_data['variety'])

# 双变量分析
sns.pairplot(iris_data, hue='variety', height=4)

# 相关性分析
plt.figure(figsize=(10, 11))
sns.heatmap(iris_data.drop('variety', axis=1).corr(), annot=True)
plt.plot()

# 箱形图
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
sns.boxplot(y='petal.width', x='variety', data=iris_data, orient='v', ax=axes[0, 0])
sns.boxplot(y='petal.length', x='variety', data=iris_data, orient='v', ax=axes[0, 1])
sns.boxplot(y='sepal.length', x='variety', data=iris_data, orient='v', ax=axes[1, 0])
sns.boxplot(y='sepal.width', x='variety', data=iris_data, orient='v', ax=axes[1, 1])

# 小提琴图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
sns.violinplot(y='petal.width', x='variety', data=iris_data, orient='v', ax=axes[0, 0], inner='quartile')
sns.violinplot(y='petal.length', x='variety', data=iris_data, orient='v', ax=axes[0, 1], inner='quartile')
sns.violinplot(y='sepal.length', x='variety', data=iris_data, orient='v', ax=axes[1, 0], inner='quartile')
sns.violinplot(y='sepal.width', x='variety', data=iris_data, orient='v', ax=axes[1, 1], inner='quartile')


# 每个特征的概率密度函数
sns.FacetGrid(iris_data, hue="variety", height=5).map(sns.histplot, "sepal.length", kde=True,stat="density", kde_kws=dict(cut=3))
plt.legend()
sns.FacetGrid(iris_data, hue="variety", height=5).map(sns.histplot, "sepal.width", kde=True,stat="density", kde_kws=dict(cut=3))
plt.legend()
sns.FacetGrid(iris_data, hue="variety", height=5).map(sns.histplot, "petal.length", kde=True,stat="density", kde_kws=dict(cut=3))
plt.legend()
sns.FacetGrid(iris_data, hue="variety", height=5).map(sns.histplot, "petal.width", kde=True,stat="density", kde_kws=dict(cut=3))
plt.legend()
plt.show()
print('Success')
