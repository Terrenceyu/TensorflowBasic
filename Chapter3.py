import tensorflow as tf
import os
from tensorflow.python.client import device_lib

## How to check and use GPU in TF
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
# if __name__ == "__main__":
#     print(device_lib.list_local_devices())


# with tf.device("/GPU:0"):
#     a = tf.constant([1.0, 2.0], shape=[1, 2])
#     b = tf.constant([5.0, 3.0], shape=[2, 1])
#
#     y = tf.matmul(a, b)
#
#     config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)
#
#     sess = tf.Session(config=config)
#
#     print(sess.run(y))


## tf.placeholder
# 有限节点高效接入大量的数据
a = tf.placeholder(tf.float32, shape=(2), name="in")
b = tf.placeholder(tf.float32, shape=(3, 2), name="in")
res = a+b

with tf.Session() as sess:
    res = sess.run(res, feed_dict={a:[1,2], b:[[1,2], [2,3], [3,4]]})
    print(res)

## tf.Variable
a = tf.Variable(tf.random_normal([3, 4], stddev=1))
# 在其他变量基础上创建新的变量
b = tf.Variable(a.initialized_value()*2.0)

# 输入常量
x = tf.constant([[1.0, 2.0]])

# seed=1 每次产生相同的随机数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 实现矩阵乘法
y = tf.matmul(tf.matmul(x, w1), w2)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(y))


## tf.get_variable() 无法创建重名的变量
# a = tf.get_variable("a", shape=[3, 4], initializer = tf.random_normal_initializer())

## tf.variable_scope()
a = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))

print(a.name)

with tf.variable_scope("s1"):
    a2 = tf.get_variable("a", [1], initializer=tf.constant_initializer(1.0))
    print(a2.name)

    with tf.variable_scope("s2"):
        a3 = tf.get_variable("a", [1])
        print(a3.name) #s1/s2/a:0


with tf.variable_scope("", reuse=True):#此时只能获取已创建的变量
    a4 = tf.get_variable("s1/s2/a", [1])
    print(a3 == a4)

## tf.name_scope()
with tf.name_scope("ns"):
    a = tf.Variable([1], name="a")
    print(a.name) #ns/a:0 受变量空间的影响

    a = tf.get_variable("b", [1])
    print(a.name) # b:0 不受变量空间的影响
