import tensorflow as tf
import numpy as np

N = 2
M = 5

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(N, 1000))  # 随机输入
# y_data = np.dot([[0.100, 0.200], [0.4, 0.5]], x_data) + 0.300

ww = np.random.rand(M, 2)
y_data = np.dot(ww, x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([M, 1]))
W = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 1000):
    sess.run(train)
    if step % 20 == 0:
        print(step, '\n', sess.run(W), '\n', sess.run(b))

print('\n', ww)

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]
