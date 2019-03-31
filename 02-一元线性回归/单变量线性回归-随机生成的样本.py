import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random, time

# ------------------------------------------步骤1：数据准备-------------------------------------------------------

# 生成随机数种子
np.random.seed(5)

# 采用np生成等差数列的方法，在【-1，1】生成100个点
x_data = np.linspace(-1, 1, 100)
# print(x_data)

# y = 2x + 1 + 噪声，噪声的维度与x_data一致
# numpy.random.randn(d0,d1,...dn)是从标准正态分布（又称u分布或0-1分布）中返回一个或多个样本值
# 实参前面加*或**，表示拆包，*表示将元组拆成一个个单独的实参
# np.random.randn(* x_data.shape)等价于np.random.randn(100)
y_data = 2 * x_data + 1.0 + np.random.randn(* x_data.shape) * 0.4
# print(y_data)

# 生成数据散点图和模型直线
# plt.scatter(x_data, y_data)
# plt.plot(x_data, 2*x_data+1, color="red", linewidth=3)
# plt.show()

#------------------------------------------步骤2：构建模型---------------------------------------------------------

# 定义训练数据的占位符，x是特征值，y是标签值
x = tf.placeholder(tf.float32, name="x")
y = tf.placeholder(tf.float32, name="y")

# 定义模型函数
def model(x, w, b):
    return tf.multiply(x, w) + b

# 创建变量tf.Variable()
# 变量初始值可以是随机数、常数，或是通过其他变量的初始值计算得到的，通常不会影响训练的结果
w = tf.Variable(1.0, name="w0")
b = tf.Variable(0.0, name="b0")

# pred是预测值节点，在w和b确定后，用来做预测用
pred = model(x, w, b)

#------------------------------------------步骤3：训练模型---------------------------------------------------------
# 设置训练超参数：迭代次数和学习率
train_epochs = 10
# 经验值学习率通常设定在0.01-0.3之间
learning_rate = 0.05
# 设置一个显示粒度的参数，表示训练多少次显示一些值（比如损失值）
display_step = 10

# 定义损失函数
# 常用损失函数有均方差MSE损失函数(也称L2损失函数)和交叉熵(cross-entropy)
loss_func = tf.reduce_mean(tf.square(y-pred))

# 定义优化器，初始化一个梯度下降（GradientDescentOptimizer）优化器，优化器是已经封装好的，可以直接调用
# 唯一的参数是学习率，优化目标是最小化损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func)

# 创建会话
# 声明会话
sess = tf.Session()
# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 迭代训练
# 过程：设置按迭代轮次，每次通过将样本逐个输入模型，进行梯度下降的优化操作
# 每轮迭代后，画出模型曲线
# 开始训练，设置初始值
step = 0
loss_list = []

for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_func], feed_dict={x: xs, y: ys})

        loss_list.append(loss)
        step += 1
        if step % display_step == 0:
            print("训练轮次%d"%(epoch+1), "训练次数%03d"%(step), "损失值%.9f"%(loss))


    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    # print(b0temp, w0temp)
    # plt.scatter(x_data, y_data)
    # plt.plot(x_data, 2 * x_data + 1, color="red", linewidth=3)
    # plt.plot(x_data, w0temp * x_data + b0temp, color="yellow",linewidth=5)
    # plt.show()
    # time.sleep(3)

# 结果可视化
# plt.scatter(x_data, y_data, label="original data")
# plt.plot(x_data, 2 * x_data + 1, label="standard line",color="red", linewidth=3)
# plt.plot(x_data, x_data * sess.run(w) + sess.run(b), label='fitted line', color='g', linewidth=5)
# plt.legend(loc=2) # 指定图例的位置
plt.plot(loss_list, "r+")
plt.show()


#-------------------------------------步骤4：用训练好的模型进行预测----------------------------------------------------
x_test = 4.55
# 以下两个求预测值的方法是等效的
predict1 = sess.run(pred, feed_dict={x: x_test})
predict2 = sess.run(w)* x_test + sess.run(b)
print("预测值1=", predict1)
print("预测值2", predict2)

target = 2 * x_test + 1
print("目标值=", target)