# 关于数据集

在竞争激烈的市场中，预测房价对于买家、卖家和房地产经纪人做出明智的决策至关重要。本项目旨在构建一个机器学习模型，根据房屋面积、位置、卧室数量和便利设施等关键特征来估算房价。

以下是每个特征的详细描述：

1. **date**：房产出售的日期。这个特征有助于了解房价的时间趋势。
2. **price**：房产的售价，单位为美元。这是我们旨在预测的目标变量。
3. **bedrooms**：房产中的卧室数量。通常情况下，卧室数量更多的房产价格会更高。
4. **bathrooms**：房产中的卫生间数量。与卧室类似，更多的卫生间可以提高房产的价值。
5. **sqft living**：居住区域的大小，以平方英尺为单位。较大的居住面积通常与更高的房产价值相关联。
6. **sqft lot**：以平方英尺为单位的土地面积。较大的地块可能会增加房产的
7. **floors**：房产中的楼层数。多层房产可能提供更多的居住空间并更具吸引力。
8. **waterfront**：一个二元指标（如果有水景则为 1，其他情况为 0）。带水景的房产通常估值更高。
9. **view**：一个从 0 到 4 的指数，表示房产视野的质量。更好的视野可能会提升房产的价值。
10. **condition**：一个从 1 到 5 的评级，用于评估房产的状况。房产状况更好的通常价值更高。
11. **sqft above**：地下室以上的房产面积。这可以帮助隔离地上空间的价值贡献。
12. **sqft basement**：地下室的使用面积。地下室可能会增加房屋价值。
13. **yr built**：房产建造年份。较旧的房产可能具有历史意义。
14. **yr renovated**：该房产上次翻新的年份。最近的翻新可以提高房产的吸引力和价值。
15. **street**：房产的街道地址。此功能可用于分析特定位置的价格趋势。
16. **city**：房产所在的城市。不同的城市有不同的市场动态。
17. **statezip**： 房产所在的州和邮编。此功能提供区域信息。
18. **country**：房产所在的 国家。虽然此数据集侧重于在澳大利亚的房产上，此功能包含在内以保持完整性。



# 1. 导入必要的库

```python
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor, plot_importance
```

> [!NOTE]
>
> - 有些库可以提前安装，我记得挺大的，像statsmodels、sklearn-learn吗，如果有anaconda好像不用安装那么多；
> - 后续也会再导入一下库，以防不知道怎么用的。



# 2. 读取数据

接下来，我们将数据集加载到环境中。如果数据集较大，可以使用duck库来进行加载。还有一件事，引号内要写清文件的绝对路径，不然可能导入失败。

```python
df = pd.read_csv('D:\\ANSTProject\\ahp\\USA Housing Dataset.csv')
```



# 3. 探索性数据分析（EDA）

EDA是数据科学中的关键步骤，旨在熟悉数据集、揭示变量间的关系并指导后续处理。通过读取数据、数据汇总、总览、缺失值和异常值分析，以及特征分析，我们可以深入了解数据的结构和质量。

```python
df.shape
df.head()
df.describe()
```

(4140, 18)

|      |        date         |   price   | bedrooms | bathrooms | sqft_living | sqft_lot | floors | waterfront | view | condition | sqft_above | sqft_basement | yr_built | yr_renovated |         street          |   city    | statezip | country |
| :--: | :-----------------: | :-------: | :------: | :-------: | :---------: | :------: | :----: | :--------: | :--: | :-------: | :--------: | :-----------: | :------: | :----------: | :---------------------: | :-------: | :------: | :-----: |
|  0   | 2014-05-09 00:00:00 | 376000.0  |   3.0    |   2.00    |    1340     |   1384   |  3.0   |     0      |  0   |     3     |    1340    |       0       |   2008   |      0       | 9245-9249 Fremont Ave N |  Seattle  | WA 98103 |   USA   |
|  1   | 2014-05-09 00:00:00 | 800000.0  |   4.0    |   3.25    |    3540     |  159430  |  2.0   |     0      |  0   |     3     |    3540    |       0       |   2007   |      0       |    33001 NE 24th St     | Carnation | WA 98014 |   USA   |
|  2   | 2014-05-09 00:00:00 | 2238888.0 |   5.0    |   6.50    |    7270     |  130017  |  2.0   |     0      |  0   |     3     |    6420    |      850      |   2010   |      0       |    7070 270th Pl SE     | Issaquah  | WA 98029 |   USA   |
|  3   | 2014-05-09 00:00:00 | 324000.0  |   3.0    |   2.25    |     998     |   904    |  2.0   |     0      |  0   |     3     |    798     |      200      |   2007   |      0       |     820 NW 95th St      |  Seattle  | WA 98117 |   USA   |
|  4   | 2014-05-10 00:00:00 | 549900.0  |   5.0    |   2.75    |    3060     |   7015   |  1.0   |     0      |  0   |     5     |    1600    |     1460      |   1979   |      0       |    10834 31st Ave SW    |  Seattle  | WA 98146 |   USA   |



## 3.1 检测缺失值

数据集经过检测，未发现缺失条目。这确保了数据的完整性，并避免了插值或删除缺失值的需要。

```python
df.isnull().sum()
```

|     date      |  0   |
| :-----------: | :--: |
|     price     |  0   |
|   bedrooms    |  0   |
|   bathrooms   |  0   |
|  sqft_living  |  0   |
|   sqft_lot    |  0   |
|    floors     |  0   |
|  waterfront   |  0   |
|     view      |  0   |
|   condition   |  0   |
|  sqft_above   |  0   |
| sqft_basement |  0   |
|   yr_built    |  0   |
| yr_renovated  |  0   |
|    street     |  0   |
|     city      |  0   |
|   statezip    |  0   |
|    country    |  0   |



## 3.2 数据清洗与预处理

- **price**：输出结果有49个样本为0，这些可能是错误或者缺失值，需要删除。
- **bedrooms**：有两个样本记录为0，是否为错误。
- **bathrooms**：有些样本的值有0.00或者0.25这样的条目，经调查可以认为属于半卫，即只有洗手池或者马桶。
- **yr renovated**：后期准备将’未翻新‘记作0。
- **street、statezip、country**：没有什么实际的意义，不影响模型的精度，准备删除。



### 3.2.1 price

```py
df['price'].value_counts().nlargest(10)
```

|   0    |  49  |
| :----: | :--: |
| 300000 |  39  |
| 400000 |  28  |
| 450000 |  27  |
| 440000 |  27  |
| 600000 |  26  |
| 350000 |  26  |
| 550000 |  25  |
| 525000 |  25  |
| 435000 |  25  |

输出结果有49个样本为0，这些可能是错误或者缺失值，需要删除。

```py
df = df[df['price'] != 0]
```

> [!CAUTION]
>
> 后面查0值同理

### 3.2.2 bedrooms

有2个样本记录为0。

```py
df = df[df['bedrooms'] != 0]
```

### 3.2.3 bathrooms

删除没有卫生间的行。

```py
df = df[df['bathrooms'] != 0]
```



# 4. 预处理后所有列的单变量分析

单变量分析考察预处理后单个变量的分布。这一步有助于理解潜在模式、检测异常值，并在进一步分析之前确保数据一致性。

## 4.1 箱线图（偏斜和异常值）

画出处理后的所有列的分布，有些存在的异常值不服从正态分布，而且有些异常值导致分布偏斜，影响统计分析和模型性能。

```py
import seaborn as sns
import matplotlib.pyplot as plt

features = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']

num_features = len(features)
rows = (num_features // 3) + (num_features % 3 > 0)
cols = 3

fig, axes = plt.subplots(rows, cols, figsize = (18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.boxplot(x = df[feature], ax = axes[i], color = sns.color_palette('viridis')[i])
    axes[i].set_title(f'Box Plot of {feature}', fontsize = 12, fontweight = 'bold')
    axes[i].set_xlabel(feature, fontsize = 11)
    axes[i].grid(axis = 'x', linestyle = '--', alpha = 0.7)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```

![Skewed Columns and Outliers](D:/XiU/Documents/Knowledge/cream/Skewed%20Columns%20and%20Outliers.png)

在箱线图中，异常值通常定义为不在 [$Q1 - 1.5 * IQR, Q3 + 1.5 * IQR$] 范围的数据点。这些点通常在箱线图中用单独的点或符号表示，并且可能不包括在箱体或须线的范围内。

## 4.2 频率分布直方图

分析具有最高计数或频率的列的分布，有助于了解主导值并识别数据集中任何重要的模式或不平衡。

```py
import seaborn as sns
import matplotlib.pyplot as plt

features = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']

num_features = len(features)
rows = (num_features // 3) + (num_features % 3 > 0)
cols = 3

fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
axes = axes.flatten()

for i, feature in enumerate(features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color=sns.color_palette("viridis")[i])
    axes[i].set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel(feature, fontsize=11)
    axes[i].set_ylabel('Frequency', fontsize=11)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```

![Distribution of Columns with the Highest Frequency](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20Columns%20with%20the%20Highest%20Frequency.png)



## 4.3使用四分位距（IQR）删除异常值

price, sqft_living, sqft_lot, sqft_above, sqft_basement 包含异常值。

```py
import pandas as pd

columns_to_filter = ['price', 'sqft_living', 'sqft_lot', 'sqft_above','sqft_basement']

def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.35)
        Q3 = df[col].quantile(0.85)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df = remove_outliers_iqr(df, columns_to_filter)

print(df.shape)
```

(3604, 18)



# 5. 去除异常值后各列的单变量分析

删除异常值后，我们再次进行单变量分析以观察更新后的分布。这一步确保数据更加均衡，并为进一步分析提供有意义的见解。

## 5.1 price分析

去除价格中的异常值后，分布几乎呈现正态分布。

```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10, 5))
sns.histplot(df['price'], bins = 30, kde = True)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

![Distribution of House Prices](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20House%20Prices.png)

```py
plt.figure(figsize = (10, 5))
common_prices = df['price'].value_counts().nlargest(10)
sns.barplot(x = common_prices.index, y = common_prices.values, palette = 'viridis')
plt.xlabel('Price')
plt.ylabel('Number of Homes')
plt.title('Most Common House Prices')
plt.xticks(rotation = 45)
plt.show()
```

![Most Common House Prices](D:/XiU/Documents/Knowledge/cream/Most%20Common%20House%20Prices-1747496941944-7.png)

从上图可以发现，大多数家庭的房价在25,000到525,000美元之间。



## 5.2 bathrooms&bedrooms分析

分析浴室和卧室的分布有助于了解住房趋势、房屋大小及其对价格的影响

```py
df['bedrooms'].value_counts()
```

|      | bedrooms |
| :--: | :------: |
| 3.0  |   1670   |
| 4.0  |   1145   |
| 2.0  |   470    |
| 5.0  |   242    |
| 6.0  |    38    |
| 1.0  |    33    |
| 7.0  |    5     |
| 8.0  |    1     |

```py
df['bathrooms'].value_counts()
```

|           | 2.50 | 1.00 | 1.75 | 2.00 | 2.25 | 1.50 | 2.75 | 3.00 | 3.50 | 3.25 | 3.75 | 4.00 | 0.75 | 4.25 | 4.50 | 1.25 | 5.25 | 5.00 |
| --------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| bathrooms | 947  | 645  | 510  | 342  | 323  | 247  | 221  | 116  | 106  | 85   | 22   | 12   | 12   | 6    | 6    | 2    | 1    | 1    |

从数据集中可以发现，大多数家庭有3到4间卧室，并且卫生间峰值在2.5间。基本可以判定其为中档住宅。



## 5.3 sqft living分析

分析 sqft_living（居住面积）有助于了解房产大小、分布及其与价格的相关性。去除异常值后的sqft_living 的分布近似服从正态分布。

```py
plt.figure(figsize=(10, 5))
sns.histplot(df['sqft_living'], bins=30, kde=True)
plt.xlabel("sqft_living")
plt.ylabel("Frequency")
plt.title("Distribution of sqft_living")
plt.show()
```

![Distribution of sqft_living](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20sqft_living.png)

从分布图中可以看出，大多数家庭住在面积在1,200到2,000平方英尺之间。



## 5.4 sqft_lot分析

分析 sqft_lot（以平方英尺为单位的土地面积）有助于了解房产分布、土地使用模式及其对房价的影响。sqft_lot 的分布接近正态分布，但右偏。为了在预处理后期的阶段获得更正态的分布，我们将应用对数转换或其他合适的技术。

```py
plt.figure(figsize=(10, 5))
sns.histplot(df['sqft_lot'], bins=30,kde=True)
plt.xlabel("sqft_lot")
plt.ylabel("Frequency")
plt.title("Distribution of sqft_lot")
plt.show()
```

![Distribution of sqft_lot](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20sqft_lot.png)

大多数房屋的占地面积在 3,000 至 10,000 平方英尺之间。



## 5.5 floors分析

分析住宅楼层数有助于理解建筑趋势、房产结构及其对价格的影响。

```py
import seaborn as sns
import matplotlib.pyplot as plt

# 统计频率
floor_counts = df['floors'].value_counts()

# 设置图形尺寸
plt.figure(figsize=(10, 5))
colors = sns.color_palette("YlGnBu", len(floor_counts))

# 画饼图（不加标签，避免重叠）
wedges, texts, autotexts = plt.pie(
    floor_counts,
    colors = colors,
    labels=None,  # 先不加文字
    autopct='%1.1f%%',
    startangle=140,
    textprops=dict(color="black", fontsize=10)  # 百分比字体白色、较小
)

# 加图例（代替标签显示）
plt.legend(wedges,
           floor_counts.index,
           title="Floors",
           loc="center left",
           bbox_to_anchor=(1, 0.5),  # 图例放右边
           fontsize=10)

plt.title("Distribution of Floors", fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

![Distribution_of_Floors](D:/XiU/Documents/Knowledge/cream/Distribution_of_Floors.png)

从饼图出发（有点丑），有48.6%的一楼住户和38.1%的二楼住户，说明住户对低楼层的偏好，可能是由于便利性和可进入性。



## 5.6 waterfront分析

检查滨水特征有助于了解其对房产价值的影响。带滨水景观的房屋通常被视为优质房产，并且可能表现出与非滨水房屋不同的定价模式。

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10,5))
sns.scatterplot(x = df['waterfront'], y = df['price'], hue = df['price'], palette = 'magma', alpha = 0.7)

plt.xlabel('Waterfront')
plt.ylabel('Price')
plt.title('House Prices vs. Waterfront')

plt.show()
```

![House Prices vs. Waterfront](D:/XiU/Documents/Knowledge/cream/House%20Prices%20vs.%20Waterfront.png)

滨水特征对房价影响很小。尽管滨水房产的价格往往会上涨，但在数据集中只有 11 个这样的条目。由于样本量较小，此功能对我们的模型预测没有显著影响。



## 5.7 view分析

更好的视野可能会提升房产的价值，但由于大多数房屋评分都是0，可以说明景观对大多数的房屋价格的影响有限。

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10, 5))
sns.scatterplot(x = df['view'], y = df['price'], hue = df['price'], palette = 'magma', alpha = 0.7)

plt.xlabel('View')
plt.ylabel('Price')
plt.title('House Prices vs. View')
plt.show()
```

![House Prices vs. View](D:/XiU/Documents/Knowledge/cream/House%20Prices%20vs.%20View.png)

虽说有评分的样本较少，但也能发现，随着景观分的增加，需求和房价往往会上升。



## 5.8 condition分析

分析房屋状况有助于了解物业维护和结构完整性如何影响价格。维护良好的房屋通常比状况较差的房屋具有更高的市场价值。

```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (10, 5))
sns.scatterplot(x = df['condition'], y = df['price'], hue = df['price'], palette = 'magma', alpha = 0.7)

plt.xlabel('Condition')
plt.ylabel('Price')
plt.title('House Prices vs. Contion')
plt.show()
```

![House Prices vs. Contion](D:/XiU/Documents/Knowledge/cream/House%20Prices%20vs.%20Contion.png)

大约有98%的房屋状况评级在3-5之间，表明大多数房屋处于较好的状况。并且通过上图可以发现，房屋状况越好，价格越高。



## 5.9 sqft_basement分析

分析 sqft_basement 有助于了解房屋中是否有地下室及其大小。拥有较大地下室的房子可能价值更高，提供额外的居住空间或存储空间。

```py
plt.figure(figsize = (10, 5))
sns.histplot(df['sqft_basement'], bins = 30, kde = True)
plt.xlabel('sqft_basement')
plt.ylabel('Frequency')
plt.title('Distribution of sqft_basement')
plt.show()
```

![Distribution of sqft_basement](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20sqft_basement.png)

大多数房屋都没有地下室。



## 5.10 sqft_above分析

分析 sqft_above（地面以上生活区域）有助于了解房屋的主要生活空间。此特征对于评估房屋面积、市场价值和建筑趋势至关重要。

```py
plt.figure(figsize = (10, 5))
sns.histplot(df['sqft_above'], bins = 30, kde = True)
plt.xlabel('sqft_above')
plt.ylabel('Frequency')
plt.title('Distribution of sqft_above')
plt.show()
```

![Distribution of sqft_above](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20sqft_above.png)

地上生活面积近似服从正态分布，并且大多少房屋地上生活面积在1,000到2,000平方英尺之间，这个范围代表了大多数房屋，表明了数据集中房屋尺寸的常见标准。



## 5.11 yr_built分析

分析 yr_built（建造年份）有助于了解房屋的年龄分布、住房发展趋势及其对房产价值的影响。较老的房屋可能具有历史意义，而较新的房屋通常提供现代设施。

```py
df['yr_built'].value_counts().nlargest(10)
#%%
plt.figure(figsize = (10, 5))
sns.histplot(df['yr_built'], bins = 30, kde = True)
plt.xlabel('yr_built')
plt.ylabel('Frequency')
plt.title('Distribution of yr_built')
plt.show()
```

![Distribution of yr_built](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20yr_built.png)



## 5.12 yr_renovated分析

分析 yr_renovated 有助于理解房屋翻新如何影响房产价值。翻新的房屋可能由于更新了功能、改善了美学和更好的结构完整性而具有更高的市场需求。

```py
plt.figure(figsize=(10, 5))
sns.histplot(df['yr_renovated'], bins=30)  
plt.title('Distribution of Year Renovated')
plt.xlabel('Year Renovated')
plt.ylabel('Count')
plt.show()
```

![Distribution of Year Renovated](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20Year%20Renovated.png)

翻新0次的样本比较多，其余年份显示不明显，故去除0次的样本。

```py
df_filtered = df[df['yr_renovated'] > 0]

plt.figure(figsize = (10, 5))
sns.histplot(df_filtered['yr_renovated'], bins = 30)
plt.title('Distribution of Year Renovated (Excluding 0)')
plt.xlabel('Year Renovated')
plt.ylabel('Count')
plt.show()
```

![Distribution of Year Renovated (Excluding 0)](D:/XiU/Documents/Knowledge/cream/Distribution%20of%20Year%20Renovated%20(Excluding%200).png)

大部分翻新时间集中在二十一世纪初。



## 5.13 street，country，statezip

这三个列都没有什么具体的意义，不影响模型的精度，并且都是美国，属于冗余的，可以删除。移除无关列有助于减少噪声并提高模型效率，从而带来更好的预测性能。

```py
df = df.drop(['street', 'country','statezip'], axis=1)
```



## 5.14 city分析

分析城市特征有助于理解位置如何影响房价和需求，不同的位置影响房地产的价值定位。

```py
city_counts = df['city'].value_counts().head(30).reset_index()
city_counts.columns = ['city', 'count']

plt.figure(figsize = (10, 6))
sns.barplot(y = city_counts['city'], x = city_counts['count'], palette= 'YlGnBu_r')
plt.xlabel('Number of Occurrences')
plt.ylabel('City')
plt.title('Most of City Distribution')
plt.show()
```

![Most of City Distribution](D:/XiU/Documents/Knowledge/cream/Most%20of%20City%20Distribution.png)

数据集中的大部分房屋位于西雅图。这表明该数据集主要代表西雅图房地产市场，这可能会影响价格、需求和房产特征的趋势。



# 6. 特征工程

```py
import numpy as np
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.show()
```

![Correlation Matrix of Numerical Features](D:/XiU/Documents/Knowledge/cream/Correlation%20Matrix%20of%20Numerical%20Features.png)

在进行进一步分析之前进行特征工程有助于通过创建更有意义的变量来提高模型性能。这一步骤涉及转换、组合或提取新特征以提高预测准确性。通过优化现有特征和创建新特征来优化数据集质量，以获得更好的洞察力和模型效率。



## 6.1 日期转换

将日期列转换为年份和月份，考察季度和时间趋势。

```py
df['date'] = pd.to_datetime(df['date'])
df['year_sold']= df['date'].dt.year
df['month_sold'] = df['date'].dt.month

df = df.drop('date', axis=1)
```



## 6.2 房屋年龄、翻新年龄和每平方英尺价格

- 房屋年龄有助于评估折旧和市场趋势。
- 改造年龄表明改造对价格和状况的影响。
- 每平方英尺价格便于比较不同房屋大小。

```py
import numpy as np

df['house_age'] = 2014 - df['yr_built']
df['renovation_age'] = np.where(df['yr_renovated'] == 0, df['house_age'], 2014 - df['yr_renovated'])
df['price_per_sqft'] = df['price'] / df['sqft_living']
```

年份特征不提供有意义的信息，因为所有记录的值都是来自 2014 年。这种缺乏变化使得它在预测建模中无帮助。

```py
df = df.drop(['year_sold'],axis=1)
```



## 6.3 价格和卧室、卫生间数量关系

```py
plt.figure(figsize = (10, 5))
sns.scatterplot(
    x = df['bedrooms'],
    y = df['bathrooms'],
    size = df['price'],
    hue = df['price'],
    sizes = (20, 180),
    palette = 'magma',
    alpha = 0.7)

plt.xlabel('Bedrooms')
plt.ylabel('Bathrooms')
plt.title('House Prices Based on Bedrooms and Bathrooms')
plt.legend(title = 'Price', loc = 'best')
plt.show()
```

![House Prices Based on Bedrooms and Bathrooms](D:/XiU/Documents/Knowledge/cream/House%20Prices%20Based%20on%20Bedrooms%20and%20Bathrooms-1747576735133-25.png)

**介绍**：关于这个散点图，是关于价格、卧室数量、卫生间数量的关系，价格由颜色的强度和大小共同表示，颜色越深价格越贵，大小越大的点价格也更高。

**趋势分析**：卧室数量、卫生间数量和房价成正比关系，越多价格越贵。



## 6.4 价格和居住面积的关系

```py
plt.figure(figsize = (10, 6))

sns.regplot(x = df['sqft_living'], y = df['price'], scatter_kws = {'alpha' : 0.4}, line_kws = {'color' : 'red'}, ci = None)

plt.xlabel('Living Area (sqft)', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.title('Relationship Between Price and Living Area', fontsize=14, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.5)

plt.show()
```

![Relationship Between Price and Living Area](D:/XiU/Documents/Knowledge/cream/Relationship%20Between%20Price%20and%20Living%20Area.png)

根据上图可以说明价格和居住面积正相关，随着居住面积（平方英尺）的增加，价格（美元）也随之上升。这与房地产的基本原理相吻合，即较大的房屋通常能卖出更高的价格。



## 6.5 按每平方英尺价格排名的前十城市

分析每平方英尺的价格有助于识别高端和实惠的市场。

- 每平方英尺最高价 → 高需求的高端地区。
- 每平方英尺最低价 → 经济实惠的住房市场。
- 每平方英尺平均价格 → 反映整体市场趋势。

较高的价格表明是黄金地段，而较低的价格则意味着是经济实惠的地区。

```py
df_stats = df.groupby('city', as_index = False).agg(
    min_price_per_sqft = ('price_per_sqft', 'min'),
    max_price_per_sqft = ('price_per_sqft', 'max'),
    avg_price_per_sqft = ('price_per_sqft', 'mean')
)

top_max_cities = df_stats.nlargest(10, 'max_price_per_sqft')
top_min_cities = df_stats.nsmallest(10, 'min_price_per_sqft')
top_avg_cities = df_stats.iloc[(df_stats['avg_price_per_sqft'] - df_stats['avg_price_per_sqft'].mean()).abs().argsort()[:10]]

def plot_top_cities(df, column, title, color):
    plt.figure(figsize = (12, 6))
    plt.barh(df['city'], df[column], color = color)
    plt.xlabel('Price per Square Foot ($)')
    plt.ylabel('City')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

plot_top_cities(top_max_cities, 'max_price_per_sqft', 'Top 10 Cities with Highest Max Price per Sqft', 'skyblue')
plot_top_cities(top_min_cities, 'min_price_per_sqft', 'Top 10 Cities with Lowest Min Price per Sqft', 'lightcoral')
plot_top_cities(top_avg_cities, 'avg_price_per_sqft', 'Top 10 Cities with Avg Price per Sqft', 'orange')
```

![Top 10 Cities with Highest Max Price per Sqft](D:/XiU/Documents/Knowledge/cream/Top%2010%20Cities%20with%20Highest%20Max%20Price%20per%20Sqft.png)

![Top 10 Cities with Lowest Min Price per Sqft](D:/XiU/Documents/Knowledge/cream/Top%2010%20Cities%20with%20Lowest%20Min%20Price%20per%20Sqft.png)

![Top 10 Cities with Avg Price per Sqft](D:/XiU/Documents/Knowledge/cream/Top%2010%20Cities%20with%20Avg%20Price%20per%20Sqft.png)

梅瑟岛、西雅图、贝尔维亚的房价排前三，塔克维拉、耶鲁点、科温顿的房价排后三。



## 6.6 各城市平均房价

分析各城市的平均房价有助于识别市场趋势和区域可负担性。

- 平均价格较高的城市表明是高端房地产市场。
- 平均价格较低则暗示有经济适用住房选择。
- 有助于投资决策和市场比较。

```py
avg_price = df.groupby('city')['price'].mean().sort_values()

plt.figure(figsize = (10, 7))
sns.set_style('whitegrid')

sns.barplot(y = avg_price.index, x = avg_price.values, palette = 'cividis')

plt.xlabel('Average Price', fontsize = 12, fontweight = 'bold')
plt.ylabel('City', fontsize = 12, fontweight = 'bold')
plt.title('Average Price of Each City', fontsize = 14, fontweight = 'bold')

sns.despine()
plt.show()
```

![Average Price of Each City](D:/XiU/Documents/Knowledge/cream/Average%20Price%20of%20Each%20City.png)

柱状图表示每个城市的平均价格，y 轴列出了城市，x 轴列出了它们相应的平均价格。柱子根据价格强度进行颜色编码。  

- 1 价格变动
  - 不同城市平均价格存在显著差异。
  - 有些城市的平均价格要高得多。

- 2 最昂贵的城市
  - 图表底部的一些城市，如克莱德山、门迪纳、梅瑟岛、贝尔维尤和贝欧阿斯村，拥有最高的平均价格。
  - 这些城市以其高端房地产、豪华住宅和理想的位置而闻名。

- 3 最低消费城市:
  - 图表顶端的城市，如斯凯科米什、阿尔戈纳、和平、塔克维拉和西塔科，平均价格最低。
  - 这些地区可能有更实惠的住房选择或较低的需求。

- 4 价格逐渐上涨：
  - 从低到高的价格过渡是平滑的，表明中间存在一系列中等价格的城市。
  - 像西雅图、雷德蒙德和基尔克兰这样的城市位于中到高价格区间，表明它们很受欢迎但并非最昂贵的。

- 5 市场趋势:
  - 价格在图表底部的急剧上升表明少数城市主导高端市场。
  - 如果价格较低的城市显示出增长的迹象，它们可能是潜在的投资领域。



## 6.7 建筑年份、出售月份、房屋年龄的关系

```py
plt.show()
#%%
columns = ['yr_built','month_sold','house_age']
plt.figure(figsize = (10, 5))

for i, column in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(x=column, y='price', data=df)
    plt.title(f'Trend of SalePrice vs {column}')

plt.tight_layout()
plt.show()
```

![Trend of SalePrice vs {column}](D:/XiU/Documents/Knowledge/cream/Trend%20of%20SalePrice%20vs%20%7Bcolumn%7D.png)

- 1 销售价格与建造年份  
  - 1950 年以前的房屋价格波动较大，在 1940 年左右有明显的价格飙升。
  - 1950 年以后，价格趋于稳定，并遵循相对一致的模式。
  - 2000 年以后的新建房屋价格往往比中期房屋略高。
  - 20 世纪初的旧房屋可能具有历史价值或优越的地理位置，这可能导致价格更高。
  - 1950 年后建造的房屋通常更经济实惠，这可能是由于建筑质量、地理位置或需求。

- 2 销售价格与销售月份的关系
  - 存在轻微的上升趋势，表明在较晚月份（尤其是 6 月和 7 月）出售的房屋价格往往更高。
  - 季节性影响：夏季月份（6 月至 7 月）需求可能增加，导致价格上涨。
  - 房地产市场趋势：买家在温暖月份更活跃，导致价格上涨。

- 3 销售价格与房屋年龄
  - 与“建造年份”趋势类似，80 年左右的房屋价格出现急剧上涨的模式。
  - 老房子（80 年以上）的价格波动性更高，这可能是由于历史意义或黄金地段所致。
  - 新近的房屋（0-40 年）价格更稳定且可预测。
  - 历史房屋效应：一些古老的房屋（80 年以上）可能因其建筑意义而价值很高。
  - 新近房屋需求：过去几十年建造的房屋可能由于现代建筑技术和供应充足而更实惠。

- 4 总体要点
  - 建于 20 世纪 40 年代和大约 80 年历史的房屋往往价格会上涨，这可能是由于历史价值。
  - 夏季月份（6 月至 7 月）售出的房屋通常价格较高，这可能是由于需求增加。
  - 较新的房屋（0-40 年）具有更可预测和稳定的定价趋势。



## 6.8 房屋状况与价格的关系

```py
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'condition', y = 'price', data = df)
plt.xlabel("Condition", fontsize=12)
plt.ylabel("Price ($)", fontsize=12)
plt.title("Boxplot of Price by Condition", fontsize=14)

plt.show()
```

![Boxplot of Price by Condition](D:/XiU/Documents/Knowledge/cream/Boxplot%20of%20Price%20by%20Condition.png)

箱型图可视化了房屋价格根据其状况评级（1 到 5）的分布情况。我们可以从数据中得出以下推断：  

- 1 价格随状况提升
  - 状况较好的房屋（状况 5）通常比状况较差的房屋具有更高的中位数价格。、
  - 这与维护良好或近期翻新的房屋通常能卖出更高价格的预期相符。

- 2  总体要点
  - 房屋状况较好的（4 级和 5 级）价格更高，且价格波动较大。
  - 状况较差的房屋（1 级和 2 级）仍处于较低的价格区间，价格变化有限。
  - 房屋翻新或装修可以显著提升房屋价值，各状况间的价格差异可见一斑。



## 6.9 地下室面积和价格的关系

```py
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

bins = [0, 500, 1000, 1500, 2000, float("inf")]
labels = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000+"]

df["basement_category"] = pd.cut(df["sqft_basement"], bins=bins, labels=labels, right=False)

df_basement = df.groupby("basement_category")["price"].mean().reset_index()

plt.figure(figsize=(10, 5))

sns.barplot(data=df_basement, x="basement_category", y="price")
sns.lineplot(data=df_basement, x="basement_category", y="price", marker='o', linewidth=2)

plt.xlabel("Basement Size Range (sqft)", fontsize=12)
plt.ylabel("Average Price ($)", fontsize=12)
plt.title("Average Price by Basement Size Range", fontsize=14)

plt.tight_layout()
plt.show()
```

![Average Price by Basement Size Range](D:/XiU/Documents/Knowledge/cream/Average%20Price%20by%20Basement%20Size%20Range.png)

根据条形图“按地下室大小范围划分的平均房价”，以下是一些关键见解：  

- 1 更大的地下室与更高的房价相关联
  - 面积在 2000 平方英尺以上的带地下室的房子平均价格最高，其次是面积在 1500-2000 平方英尺之间的房子。
  - 这表明买家愿意为拥有更大地下室的房子支付溢价。

- 2 地下室面积越大，价格越高，但增长速度较慢
  - 从面积较小的地下室（0-1500 平方英尺）到中等大小的地下室，价格跳跃显著。
  - 然而，在 1500-2000 平方英尺和 2000+平方英尺之间的价格上涨幅度较小，表明超过一定面积后，地下室对价格的影响会减弱。

- 3 潜在买家偏好
  - 如果大多数买家优先考虑可用的地下室空间，那么拥有 1000-1500 平方英尺地下室的房子在价格和大小之间提供了很好的平衡。
  - 豪华买家可能特别寻找 2000 平方英尺以上的地下室，这使得这些房子的价格更高。

- 4 投资角度
  - 如果你是一个房屋投资者或卖家，增加或扩大地下室空间（尤其是从 500 平方英尺增加到 1500 平方英尺以上）可以显著提高房屋价值。
  - 然而，超过 2000 平方英尺的扩张可能会使价格增值产生递减回报。



## 6.10 翻新时间和平均房价的关系

```py
df_renovated = df[df["yr_renovated"] > 0]
df_renovated = df_renovated.groupby("yr_renovated")["price"].mean().reset_index()
df_renovated = df_renovated.sort_values(by="price", ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(data=df_renovated, x="yr_renovated", y="price")
plt.xlabel("Year Renovated", fontsize=12)
plt.ylabel("Average Price ($)", fontsize=12)
plt.title("Average Home Prices by Renovation Year", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig(r'D:\数据分析\ANSTProject\ahp\Average Home Prices by Renovation Year.png')
plt.show()
```

![Average Home Prices by Renovation Year](D:/XiU/Documents/Knowledge/cream/Average%20Home%20Prices%20by%20Renovation%20Year.png)

对其重新定义：0 → 从未翻新，1 → 已翻新

- 近一半的房子从未进行过翻新，这表明翻新状况可能会影响价格和需求。
- 这项功能也将有助于编码，使其更容易集成到机器学习模型中。

```py
df['renovation_status'] = df['yr_renovated'].apply(lambda x: '0' if x == 0 else '1')
df = df.drop('yr_renovated', axis=1)
df['renovation_status'].value_counts()
```

|      | renovation_status |
| ---- | :---------------: |
| 0    |       2126        |
| 1    |       1478        |



## 6.11 地下室与价格的关系

如果 sqft_basement = 0，这栋房子没有地下室。如果 sqft_basement > 0，则该房屋有地下室。这种区分有助于分析地下室对房价和整体房产价值的影响。这也将有助于编码

```py
df['basement_status'] = df['sqft_basement'].apply(lambda x: '0' if x == 0 else '1')
df = df.drop('sqft_basement', axis=1)
df['basement_status'].value_counts()
```

|      | basement_status |
| ---- | :-------------: |
| 0    |      2126       |
| 1    |      1478       |

有一半的房屋没有地下室，这说明是否有地下室可能会影响到房价。



# 导出导入文件，准备进行建模

```py
df.to_csv('cleaned_data.csv', index=False)
```

```py
df = pd.read_csv('cleaned_data.csv')
```

```py
df.columns
```

Index(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_above', 'yr_built', 'city',
       'month_sold', 'house_age', 'renovation_age', 'price_per_sqft',
       'basement_category', 'renovation_status', 'basement_status'],
      dtype='object')

我们删除没价格没有关系的列，减少模型的复杂度。

```py
df = df.drop(['house_age', 'month_sold', 'basement_category', 'city'], axis=1)
```



# 7. 多元线性回归

本节介绍如何利用多元线性回归方法对美国房价进行预测。我们将首先检测解释变量之间的多重共线性，再构建线性回归模型，并通过模型评估和可视化展示预测效果，最后介绍基于 XGBoost 算法的进一步建模和预测步骤。

## 7.1 多重共线性检验

在使用多元线性回归时，确保各自变量间不存在严重共线性问题十分重要，因为高度相关的变量可能导致模型估计不稳定。这里我们采用方差膨胀因子（VIF）来检测多重共线性。通常情况下，当某变量的 VIF 大于 5 时，说明该变量与其他解释变量之间存在较强的共线性；在这种情况下，通过剔除或合并部分变量可以提升模型的可靠性。需要注意的是，此检验主要适用于数值型变量，因此应避免对虚拟变量进行 VIF 计算。

```py
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

dff = df.drop(columns=['price'])
X = sm.add_constant(dff)

vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data = vif_data[vif_data["feature"] != "const"]
vif_data = vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

vif_data.head(14)
```

|      |      feature      |    VIF    |
| :--: | :---------------: | :-------: |
|  0   |    sqft_above     | 15.355609 |
|  1   |    sqft_living    | 15.281073 |
|  2   |  basement_status  | 3.963955  |
|  3   |     bathrooms     | 3.147647  |
|  4   |     yr_built      | 2.662540  |
|  5   |      floors       | 2.353004  |
|  6   |  renovation_age   | 1.885062  |
|  7   |     bedrooms      | 1.761993  |
|  8   |     condition     | 1.623041  |
|  9   | renovation_status | 1.622671  |
|  10  |     sqft_lot      | 1.400107  |
|  11  |  price_per_sqft   | 1.312627  |
|  12  |       view        | 1.241772  |
|  13  |    waterfront     | 1.107334  |

上述结果显示，`sqft_above` 和 `sqft_living` 的 VIF 值均超过了 15，表明二者之间存在较强的共线性。尽管在某些模型（如 XGBoost）中可以保留这两个变量，但在传统线性回归建模时通常需要剔除其中一个以降低多重共线性对模型的影响。



## 7.2 线性回归

### 数据预处理与模型构建

针对右偏的数据分布，我们对 `sqft_lot` 变量进行了对数转换；同时，为降低多重共线性的影响，在进行建模时剔除了 `price` 与 `sqft_above` 变量（后续在 XGBoost 中会考虑保留）。随后，我们将数据集划分为训练集和测试集，并利用 scikit-learn 的 `LinearRegression` 模型进行拟合和预测。

```py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import numpy as np

df = df.copy()
df['sqft_lot'] = np.log(df['sqft_lot'])

X = df.drop(columns=['price', 'sqft_above'])
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {round(mse, 3)}')
print(f'R^2 Score: {round(r2, 3)}')

coefficients = model.coef_
print(f"回归系数: {list(map(lambda x: round(x, 3), coefficients))}")
intercept = model.intercept_
print(f"截距: {round(intercept, 3)}")

X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

residuals = y_test - y_pred

within_range = (y_pred >= 0.8 * y_test) & (y_pred <= 1.2 * y_test)
proportion_within_range = np.mean(within_range)
print(f'预测值在真实值的0.8到1.2倍之间的比例: {round(proportion_within_range, 3)}')

within_range = (y_pred >= 0.7 * y_test) & (y_pred <= 1.3 * y_test)
proportion_within_range = np.mean(within_range)
print(f'预测值在真实值的0.7到1.3倍之间的比例: {round(proportion_within_range, 3)}')
```

Mean Squared Error: 4419506157.453
R^2 Score: 0.918
回归系数: [5069.542, 15773.773, 235.981, 15256.377, 11848.938, -27591.362, 11740.255, 4384.029, -107.762, 46.892, 1729.495, 181.772, 14321.973]
截距: -435516.592
预测值在真实值的0.8到1.2倍之间的比例: 0.859
预测值在真实值的0.7到1.3倍之间的比例: 0.908

```py
print(model_sm.summary())
```

|        指标        |                数值                 |
| :----------------: | :---------------------------------: |
|     R-squared      | **0.907** （解释了90.7%的房价方差） |
|   Adj. R-squared   |                0.906                |
|    F-statistic     |       2149.0 （模型整体显著）       |
| Prob (F-statistic) |                0.000                |
|        AIC         |               72890.0               |
|        BIC         |               72980.0               |

统计结果中，模型的判定系数（R-squared）为 **0.907**，说明该模型能够解释 90.7% 的房价波动；此外，F 值显著（F-statistic = 2149.0，P 值几乎为 0），表明整体模型非常显著。在剔除 P 值大于 0.05 的解释变量（如 `waterfront`、`renovation_age`、`renovation_status` 和 `yr_built`）后，模型的判定系数依然保持在 0.907 左右，从而印证了模型的鲁棒性和预测准确度。

下面是剔除'waterfront' , 'renovation_age', 'renovation_status', 'yr_built' P值大于0.05后的回归结果。

![image-20250518231919982](D:/XiU/Documents/Knowledge/cream/image-20250518231919982.png)

判定系数高达0.907，并且修正后的判定系数0.907，和原判定系数相同，且落入真值的预测值有0.908，精度较高。

接下来，我们利用散点图展示了实际房价与预测房价的对比情况：

```py
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual House Prices')
plt.grid(True)
plt.show()
```

![aa](D:/XiU/Documents/Knowledge/cream/aa-1747581667624-2.png)



## 7.3 XGBOOT模型

为进一步提升预测性能，我们构建了基于 XGBoost 的房价预测模型。XGBoost 能够更好地刻画特征之间的非线性关系和复杂交互作用，常在回归任务中展现出卓越性能。

#### 数据预处理与管道构建

同样对 `sqft_lot` 使用对数转换后，我们将数据集中的数值型和分类变量进行区分，并构建了数据预处理 pipeline，其中：

- 数值型变量采用 `StandardScaler` 进行标准化；
- 分类变量采用 `OneHotEncoder` 编码处理，确保遇到未知类别时不报错。

随后，我们将数据划分为训练集和测试集，利用 GridSearchCV 对 XGBoost 模型超参数进行调优，构建最终的预测流水线。

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor, plot_importance

# 特征准备
df = df.copy()
df['sqft_lot'] = np.log(df['sqft_lot'])
X = df.drop(columns=['price'])
y = df['price']

# 区分特征类型
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# 构建预处理 pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 XGBoost 模型
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# 构建总 pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])
```

利用如下超参数网格进行调参，再通过 3 折交叉验证选出最佳模型：

```py
param_grid = {
    'model__n_estimators': [50, 100],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# 网格搜索调参
grid_search = GridSearchCV(
    pipeline, param_grid, scoring='r2',
    cv=3, verbose=1, n_jobs=1
)

grid_search.fit(X_train, y_train)
```

![image-20250519010019827](D:/XiU/Documents/Knowledge/cream/image-20250519010019827.png)

```py
y_pred = grid_search.best_estimator_.predict(X_test)

print(f"Model R^2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
```

Model R^2 Score: 0.9972

MAE: 7604.38

RMSE: 12355.44

```py
# 如果之前没分 train/test，可以用交叉验证：
from sklearn.model_selection import cross_val_score

scores = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='r2')
print("Cross-validated R² scores:", scores)
print("Mean R²:", scores.mean())
```

Cross-validated R² scores: [0.99673845 0.99647982 0.99673527 0.99687792 0.99464211]
Mean R²: 0.9962947164907616

同时，通过 5 折交叉验证，模型的平均 R² 达到了约 0.9963，表明整体预测效果极为优异。

我们进一步提取最佳模型中 XGBoost 部分的特征重要性信息，并绘制出前20个最重要特征的条形图，以直观展示哪些变量对房价影响最大。

```py
best_model = grid_search.best_estimator_.named_steps['model']

# 获取原始特征名（不包括 OneHot）
raw_feature_names = X.columns.tolist()

# 为 XGBoost 模型绑定特征名（仅对 plot_importance 有效）
booster = best_model.get_booster()
booster.feature_names = raw_feature_names  # 注意：此处 X 不能 OneHot

# 绘图
plt.figure(figsize=(10, 6))
plot_importance(booster, max_num_features=20, height=0.4)
plt.title("XGBoost Feature Importance")
plt.savefig("XGBoost Feature Importance.png")
plt.show()
```

![XGBoost Feature Importance](D:/XiU/Documents/Knowledge/cream/XGBoost%20Feature%20Importance-1747639193331-2.png)

接下来，我们将训练好的模型保存至本地，以便后续部署或进一步应用。同时，通过散点图展示实际房价与预测房价的关系：

```py
# 保存模型
with open("xgboost_house_price_model.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)

# 实际 vs 预测可视化
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label="Predicted vs. Actual")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction", linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.legend()
plt.savefig(r"D:\数据分析\ANSTProject\ahp\Actual vs. Predicted House Prices.png")
plt.show()
```

![Actual vs. Predicted House Prices](D:/XiU/Documents/Knowledge/cream/Actual%20vs.%20Predicted%20House%20Prices.png)



## 7.4 预测房价

在获得最佳模型后，我们编写了一个预测函数 `predict_house_price`，能对单个样本进行房价预测。函数首先将输入的字典转换为 DataFrame，并按训练集的特征顺序调整列的顺序，然后利用最佳模型进行预测。示例中，对给定变量的预测结果约为 395,183.72 美元。

```py
def predict_house_price(sample):
    sample_df = pd.DataFrame([sample])
    sample_df = sample_df[X_train.columns] 
    predicted_log_price = grid_search.best_estimator_.predict(sample_df)
    predicted_price = np.expm1(predicted_log_price) if np.any(y_train <= 0) else predicted_log_price
    return predicted_price[0]

sample_house = {
    'yr_built': 100,
    'bedrooms': 5,
    'bathrooms': 2,
    'sqft_living': 500,
    'sqft_lot': 260,
    'floors': 2,
    'waterfront': 0,
    'view': 24,
    'condition': 30,
    'sqft_above': 151,
    'renovation_age': 92,
    'price_per_sqft': 952,
    'renovation_status': 1,
    'basement_status': 1
}

print(f"Predicted Price: ${predict_house_price(sample_house):,.2f}")
```

在上述变量值的预测中，结果为395,183.72美元。
