"""
    我们来用SK-Learn写一个简单的线性回归。
    数据存在一个名为 data.txt 的文件中，该文件的格式类似csv，由英文逗号分隔，第一列是输入特征，第二列是对应的输出值
    你需要首先读取文件，并将数据分别保存到名为 X 和 y 的两个列表中
"""
import numpy as np

X = list()
y = list()

### STEP 1 读取和保存数据
with open("data.txt") as f:
    lines = f.readlines()
    for line in lines:
        xt, yt = line.strip().split(",")
        X.append(float(xt))
        y.append(float(yt))
### STEP 1 END
"""
    接下来我们构造一个简单的训练集和测试集。
    我们以80%为界，X 和 y 的前80%数据为训练集，后20%为测试集
    请将数据放入到对应的变量名中，格式为Numpy Array
    为了便于后续处理，请将训练集和测试集的 X 竖置（变形为 N×1 的向量）
"""
# 训练集的 X 和 y 请分别放到 X_train 和 y_train中
X_train = np.array([])
y_train = np.array([])

# 测试集的 X 和 y 请分别放到 X_test 和 y_test 中
X_test = np.array([])
y_test = np.array([])

### STEP 2 构造训练集 / 测试集
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train = np.array(X[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])

X_test = np.array(X[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])
### STEP 2 END

"""
    接下来我们用SK-Learn中的线性回归模型拟合训练数据
    请将拟合好回归器对象赋给 linear_regressor
"""

from sklearn import linear_model

linear_regressor = None

### STEP 3 构造线性回归器并拟合数据
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
### STEP 3 END

"""
    好极了！现在我们需要将训练出的回归模型用图表展示出来。
    请完善代码，使得最后一行plt.show 可输出符合如下条件的图：
    - 该图以 X, y 为 坐标轴，将训练集中的数据点用绿色圆点标出
    - 将 linear_regressor 的训练结果以黑色、线宽为4的直线标出
    - 以 Training data 作为图表标题
"""

import matplotlib.pyplot as plt

### STEP 4 可视化
y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()
### STEP 4 END

plt.show()


"""
    恭喜！你已经完成本测试的所有任务，请提交你的脚本。
"""
