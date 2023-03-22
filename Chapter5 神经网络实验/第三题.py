import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import ListedColormap



from sklearn.datasets import make_moons
X, y = make_moons(n_samples = 2000, noise = 0.3, random_state=0)



plt.figure(figsize = (6, 6))
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cm_bright, edgecolors = 'k')



from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X, y, test_size = 0.4, random_state = 32)
trainY = trainY
testY = testY



trainX.shape, trainY.shape, testX.shape, testY.shape



from sklearn.preprocessing import StandardScaler
s = StandardScaler()
trainX = s.fit_transform(trainX)
testX = s.transform(testX)



def initialize(m):
    '''
    初始化参数W和参数b
    
    Returns
    ----------
    W: np.ndarray, shape = (m, )，参数W
    
    b: np.ndarray, shape = (1, )，参数b
    
    '''
    np.random.seed(32)
    W = np.random.normal(size = (m, )) * 0.01
    b = np.zeros((1, ))
    return W, b


Wt, bt = initialize(trainX.shape[1])
print(Wt.shape) # (2,)
print(bt.shape) # (1,)


def linear_combination(X, W, b):
    '''
    完成Z = XW + b的计算
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，输入的数据
    
    W: np.ndarray, shape = (m, )，权重
    
    b: np.ndarray, shape = (1, )，偏置
    
    Returns
    ----------
    Z: np.ndarray, shape = (n, )，线性组合后的值
    
    '''
    
    # YOUR CODE HERE
    
    Z = np.dot(X, W) + b 
    
    return Z


def my_sigmoid(x):
    '''
    simgoid 1 / (1 + exp(-x))
    
    Parameters
    ----------
    X: np.ndarray, 待激活的值
    
    '''
    # YOUR CODE HERE
    
    activations = 1 / (1 + np.exp(-x))
    
    return activations


Wt, bt = initialize(trainX.shape[1])
Zt = linear_combination(trainX, Wt, bt)
my_sigmoid(Zt).mean() # 0.49999



np.exp(1e56)

my_sigmoid(np.array([-1e56]))



from scipy.special import expit



def sigmoid(X):
    return expit(X)


def forward(X, W, b):
    '''
    完成输入矩阵X到最后激活后的预测值y_pred的计算过程
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m)，数据，一行一个样本，一列一个特征
    
    W: np.ndarray, shape = (m, )，权重
    
    b: np.ndarray, shape = (1, )，偏置
    
    Returns
    ----------
    y_pred: np.ndarray, shape = (n, )，模型对每个样本的预测值
    
    '''
    # 求Z
    # YOUR CODE HERE
    
    Z = linear_combination(X, W, b)
    
    # 求激活后的预测值
    # YOUR CODE HERE

    y_pred = sigmoid(Z)
    
    return y_pred


# 测试样例
Wt, bt = initialize(trainX.shape[1])
forward(trainX, Wt, bt).mean() # 0.4999(没有四舍五入)


# 接下来完成损失函数的编写，我们使用的是对数损失，这里需要注意的一个问题是：
# 
# $$
# \mathrm{loss}(y, \hat{y}) = - \frac{1}{n}[ y \log{\hat{y}} + (1 - y) \log{(1 - \hat{y})}]
# $$
# 
# 在这个对数损失中，$\hat{y}$中不能有$0$和$1$，如果有$0$，那么损失函数中的前半部分，$\log{0}$就会出错，如果有$1$，那么后半部分$\log{(1-1)}$就会出错。
# 
# 所以我们要先将$\hat{y}$中的$0$和$1$改变一下，把$0$变成一个比较小但是大于$0$的数，把$1$变成小于$1$但是足够大的数。使用`numpy.clip`函数就可以作到这点。


def logloss(y_true, y_pred):
    '''
    给定真值y，预测值y_hat，计算对数损失并返回
    
    Parameters
    ----------
    y_true: np.ndarray, shape = (n, ), 真值
    
    y_pred: np.ndarray, shape = (n, )，预测值
    
    Returns
    ----------
    loss: float, 损失值
    
    '''
    # 下面这句话会把y_pred里面小于1e-10的数变成1e-10，大于1 - 1e-10的数变成1 - 1e-10
    y_hat = np.clip(y_pred, 1e-10, 1 - 1e-10)
    
    # 求解对数损失
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))                                     # YOUR CODE HERE
    
    return loss



# 测试样例
Wt, bt = initialize(trainX.shape[1])
logloss(trainY, forward(trainX, Wt, bt)) # 0.69740


# ### 3.3 反向传播

# 我们接下来要完成损失函数对参数的偏导数的计算

def compute_gradient(y_true, y_pred, X):
    '''
    给定预测值y_pred，真值y_true，传入的输入数据X，计算损失函数对参数W的偏导数的导数值dW，以及对b的偏导数的导数值db
    
    Parameters
    ----------
    y_true: np.ndarray, shape = (n, ), 真值
    
    y_pred: np.ndarray, shape = (n, )，预测值
    
    X: np.ndarray, shape = (n, m)，数据，一行一个样本，一列一个特征
    
    Returns
    ----------
    dW: np.ndarray, shape = (m, ), 损失函数对参数W的偏导数
    
    db: float, 损失函数对参数b的偏导数
    
    '''
    # 求损失函数对参数W的偏导数的导数值
    # YOUR CODE HERE

    dW = -np.sum((y_true - y_pred) * X.T, axis =1)  / X.shape[0]
    
    # 求损失函数对参数b的偏导数的导数值
    # YOUR CODE HERE
    db = np.sum(y_pred - y_true) / X.shape[0]
    
    return dW, db



# 测试样例
Wt, bt = initialize(trainX.shape[1])
dWt, dbt = compute_gradient(trainY, forward(trainX, Wt, bt), trainX)
print(dWt.shape) # (2, )
print(dWt.sum()) # 0.04625
print(dbt)       # 0.00999


# ### 3.4 参数更新


def update(W, b, dW, db, learning_rate):
    '''
    梯度下降，给定参数W，参数b，以及损失函数对他们的偏导数，使用梯度下降更新参数W和参数b
    
    Parameters
    ----------
    W: np.ndarray, shape = (m, )，参数W
    
    b: np.ndarray, shape = (1, )，参数b
    
    dW: np.ndarray, shape = (m, ), 损失函数对参数W的偏导数
    
    db: float, 损失函数对参数b的偏导数
    
    learning_rate, float，学习率
    
    '''
    # 对参数W进行更新
    W -= learning_rate * dW
    
    # 对参数b进行更新
    # YOUR CODE HERE
    b += learning_rate * db


# 测试样例
Wt, bt = initialize(trainX.shape[1])
print(Wt)  # [-0.00348894  0.00983703]
print(bt)  # [ 0.]
print()

dWt, dbt = compute_gradient(trainY, forward(trainX, Wt, bt), trainX)
print(dWt) # [-0.28650366  0.33276308]
print(dbt) # 0.00999999939463
print()

update(Wt, bt, dWt, dbt, 0.01)
print(Wt)  # [-0.00062391  0.0065094 ]
print(bt)  # [ -9.99999939e-05]

def backward(y_true, y_pred, X, W, b, learning_rate):
    '''
    反向传播，包含了计算损失函数对各个参数的偏导数的过程，以及梯度下降更新参数的过程
    
    Parameters
    ----------
    y_true: np.ndarray, shape = (n, ), 真值
    
    y_pred: np.ndarray, shape = (n, )，预测值
    
    X: np.ndarray, shape = (n, m)，数据，一行一个样本，一列一个特征
    
    W: np.ndarray, shape = (m, )，参数W
    
    b: np.ndarray, shape = (1, )，参数b
    
    dW: np.ndarray, shape = (m, ), 损失函数对参数W的偏导数
    
    db: float, 损失函数对参数b的偏导数
    
    learning_rate, float，学习率
    
    '''
    # 求参数W和参数b的梯度
    dW, db = compute_gradient(y_true, y_pred, X)
    
    # 梯度下降
    update(W, b, dW, db, learning_rate)


# 测试样例
Wt, bt = initialize(trainX.shape[1])
y_predt = forward(trainX, Wt, bt)
loss_1 = logloss(trainY, y_predt)
print(loss_1)                             # 0.697403529518

backward(trainY, y_predt, trainX, Wt, bt, 0.01)

y_predt = forward(trainX, Wt, bt)
loss_2 = logloss(trainY, y_predt)
print(loss_2)                             # 0.695477626714


# ## 4. 训练函数的编写




def train(trainX, trainY, testX, testY, W, b, epochs, learning_rate = 0.01, verbose = False):
    '''
    训练，我们要迭代epochs次，每次迭代的过程中，做一次前向传播和一次反向传播
    同时记录训练集和测试集上的损失值，后面画图用
    
    Parameters
    ----------
    trainX: np.ndarray, shape = (n, m), 训练集
    
    trainY: np.ndarray, shape = (n, ), 训练集标记
    
    testX: np.ndarray, shape = (n_test, m)，测试集
    
    testY: np.ndarray, shape = (n_test, )，测试集的标记
    
    W: np.ndarray, shape = (m, )，参数W
    
    b: np.ndarray, shape = (1, )，参数b
    
    epochs: int, 要迭代的轮数
    
    learning_rate: float, default 0.01，学习率
    
    verbose: boolean, default False，是否打印损失值
    
    Returns
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值
    
    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值
    
    '''
    
    training_loss_list = []
    testing_loss_list = []
    
    for i in range(epochs):
        
        # 计算训练集前向传播得到的预测值
        # YOUR CODE HERE
        train_y_pred = linear_combination(trainX, W, b)
        # 计算当前训练集的损失值
        # YOUR CODE HERE
        training_loss = logloss(trainY, train_y_pred)

        # 计算测试集前向传播得到的预测值
        # YOUR CODE HERE
        test_y_pred = linear_combination(testX, W, b)
        # 计算当前测试集的损失值
        # YOUR CODE HERE
        testing_loss = logloss(testY, test_y_pred)
        
        if verbose == True:
            print('epoch %s, training loss:%s'%(i + 1, training_loss))
            print('epoch %s, testing loss:%s'%(i + 1, testing_loss))
            print()
        
        # 保存损失值
        training_loss_list.append(training_loss)
        testing_loss_list.append(testing_loss)
        
        # 反向传播更新参数
        # YOUR CODE HERE
        backward(trainY, train_y_pred, trainX, W, b, learning_rate)

    
    return training_loss_list, testing_loss_list


# 测试样例
Wt, bt = initialize(trainX.shape[1])
training_loss_list, testing_loss_list = train(trainX, trainY, testX, testY, Wt, bt, 2, 0.1)
print(training_loss_list)  # [0.69740352951773121, 0.67843729060725722]
print(testing_loss_list)   # [0.69743661286103986, 0.67880126235588389]


# ## 5. 绘制模型损失值变化曲线

# In[ ]:


def plot_loss_curve(training_loss_list, testing_loss_list):
    '''
    绘制损失值变化曲线
    
    Parameters
    ----------
    training_loss_list: list(float)，每迭代一次之后，训练集上的损失值
    
    testing_loss_list: list(float)，每迭代一次之后，测试集上的损失值
    
    '''
    plt.figure(figsize = (10, 6))
    plt.plot(training_loss_list, label = 'training loss')
    plt.plot(testing_loss_list, label = 'testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()


# ## 6. 预测

# 接下来编写一个预测的函数，事实上，$\rm sigmoid$输出的是当前这个样本为正例的概率，也就是说，这个输出值是一个0到1的值，一般我们将大于0.5的值变成1，小于0.5的值变成0，也就是说，如果当前输出的概率值大于0.5，那我们认为这个样本的类别就是1，否则就是0，这样输出的就是类标了。

# In[ ]:

def predict(X, W, b):
    '''
    预测，调用forward函数完成神经网络对输入X的计算，然后完成类别的划分，大于0.5的变为1，小于等于0.5的变为0
    
    Parameters
    ----------
    X: np.ndarray, shape = (n, m), 训练集
    
    W: np.ndarray, shape = (m, 1)，参数W
    
    b: np.ndarray, shape = (1, )，参数b
    
    Returns
    ----------
    prediction: np.ndarray, shape = (n, 1)，预测的标记
    
    '''
    
    # YOUR CODE HERE
    return prediction


# In[ ]:


# 测试样例
from sklearn.metrics import accuracy_score
Wt, bt = initialize(trainX.shape[1])
predictiont = predict(testX, Wt, bt)
accuracy_score(testY, predictiont)  # 0.16250000000000001


# ## 7. 训练一个神经网络

# 我们的学习率是0.01，迭代200轮

# In[ ]:


W, b = initialize(trainX.shape[1])
training_loss_list, testing_loss_list = train(trainX, trainY, testX, testY, W, b, 200, 0.01)


# 计算测试集精度

# In[ ]:


prediction = predict(testX, W, b)
accuracy_score(testY, prediction)  # 0.83625000000000005


# 绘制损失值变化曲线

# In[ ]:


plot_loss_curve(training_loss_list, testing_loss_list)


# # 课后部分：初始化新的参数，学习率和迭代轮数按下表设置，绘制其训练集和测试集损失值的变化曲线，完成表格内精度的填写，并结合参数学习率对曲线变化及训练过程进行分析

# ###### 双击此处填写
# 
# 学习率|迭代轮数|测试集精度
# -|-|-
# 0.0001|200|
# 0.1|1000|

# In[ ]:


# YOUR CODE HERE





# In[ ]:


# YOUR CODE HERE





# In[ ]:




