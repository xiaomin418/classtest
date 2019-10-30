# https://blog.csdn.net/ningyanggege/article/details/83895647
import tensorflow as tf
import numpy as np

#创建一个神经网络层
def add_layer(input,in_size,out_size,activation_function=None):
    with tf.variable_scope("foo"):
        Weight = tf.Variable(tf.random_normal([in_size, out_size]),name='layer1')
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='b1')
    W_mul_x_plus_b=tf.matmul(input,Weight) + biases
    #根据是否有激活函数
    if activation_function == None:
        output=W_mul_x_plus_b
    else:
        output=activation_function(W_mul_x_plus_b)
    return output

def add_layer_decode(input,in_size,out_size,activation_function=None):
    with tf.variable_scope("foo",reuse=True):
        Weight = tf.Variable(tf.random_normal([out_size, in_size]), name='layer1')
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b1')
    W_mul_x_plus_b=tf.matmul(input,tf.transpose(Weight)) + biases
    #根据是否有激活函数
    if activation_function == None:
        output=W_mul_x_plus_b
    else:
        output=activation_function(W_mul_x_plus_b)
    return output

x_data = [[0,0,1,3,5,6, 0, 0], [0,0,1, 4, 1,9,0,0], [0,0,4,3,3,2,0,0], [0,0,2,4,9,1,0,0],
                                 [0, 0, 8,7,5,4,0,0], [0,0,2,4,5,6,0,0], [0, 0, 1,9,4,6,0,0],[0, 0, 7,7,3,2,0,0]]
y_data = [[0,0,1,3,5,6, 0, 0], [0,0,1, 4, 1,9,0,0], [0,0,4,3,3,2,0,0], [0,0,2,4,9,1,0,0],
                                 [0, 0, 8,7,5,4,0,0], [0,0,2,4,5,6,0,0], [0, 0, 1,9,4,6,0,0],[0, 0, 7,7,3,2,0,0]]
x_test= [[0, 0, 3,5,6,7,0,0]]
y_test = [[0, 0, 3,5,6,7,0,0]]
xs=tf.placeholder(tf.float32,[None,8])
ys=tf.placeholder(tf.float32,[None,8])
tf.get_variable_scope().reuse_variables()
#定义一个隐藏层
hidden_layer1=add_layer(xs,8,4,activation_function=tf.nn.sigmoid)
#定义一个输出层
prediction=add_layer_decode(hidden_layer1,4,8,activation_function= tf.nn.sigmoid)
# 求解神经网络参数
# 1.定义损失函数
loss = tf.reduce_sum(tf.square(ys - prediction),1)
# 2.定义训练过程
train_step = tf.train.GradientDescentOptimizer(1).minimize(loss)  # 梯度下降法使误差最小，学习率为0.1

init = tf.global_variables_initializer() # 变量初始化
sess = tf.Session()
sess.run(init)  # 执行初始化
# 3.进行训练
for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})  # 训练
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))  # 输出当前loss
print(sess.run(prediction,feed_dict={xs: x_test}))

# 关闭sess
sess.close()
