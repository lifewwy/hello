import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

N = 3
M = 5
miniBatchSize = 50
nData = 1000

# 使用 NumPy 生成假数据(phony data), 总共 nData 个点.
x_data = np.float32(np.random.rand(N, nData))  # 随机输入


ww = np.random.rand(M, N)
y_data = np.dot(ww, x_data) + 0.300

x_ = tf.placeholder(tf.float32, [N, None])
y_ = tf.placeholder(tf.float32, [M, None])

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([M, 1]))
W = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0))
y = tf.matmul(W, x_) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 2000):
    ind = np.random.randint(0, nData, miniBatchSize)
    sess.run(train, feed_dict={x_: x_data[:, ind], y_: y_data[:, ind]})
    if step % 20 == 0:
        print(step, '\n', sess.run(W), '\n\n', sess.run(b))

print('\n', ww, '\n')
print('\n', np.linalg.norm(ww - sess.run(W)), '\n')

'''
plt.plot(x_data[1, :])
plt.show() '''

# print('\n', np.random.randint(0, 1000, 30), '\n')
# print(x_data[:, np.random.randint(0, 1000, 30)])


